import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import numpy as np
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class InformativenessNormCorrelation(nn.Module):
    """
    Spearman's rank correlation between the sentences' vector norm and sentence informativeness
    """
    
    def __init__(self, model_args, data_args):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.model_args = model_args
        self.data_args = data_args
    
    def forward(self, pooler_output, rank=None, group=None):
        loss_norm_informativeness = 0
        if rank is not None and group is not None:
            output_norm = torch.norm(pooler_output, dim=-1)
            n1, n2 = output_norm[:, 0], output_norm[:, 1]
            rank = rank.squeeze()[:, 0]
            group = group.squeeze()[:, 0]
            for n in [n1, n2]:
                # # use correlation
                # norm_rank_group = torch.stack([n, rank, group], dim=1)
                # unique_g = torch.unique(group)
                # result = torch.empty((unique_g.shape[0], 2))
                # i = 0
                # for g in unique_g:
                #     n_g = norm_rank_group[group == g]
                #     nr = n_g[:, 0].sort().indices
                #     r = n_g[:, 1]
                #     corr = self.spearmans_rank_corr(nr, r)
                #     result[i, :] = torch.Tensor([corr, r.shape[0]])
                #     i += 1
                # result[:, 0][torch.isnan(result[:, 0])] = 0 # replace nan to 0
                # result[:, 0] = 1- result[:, 0] # each group's correlation should be 1, so the loss is (1 - correlation)
                # loss_temp = (result[:, 0] * result[:, 1]).sum() / result[:, 1].sum()
                # loss_norm_informativeness += loss_temp
                
                # # use MSELoss
                # norm_rank_group = torch.stack([n, rank, group], dim=1)
                # unique_g = torch.unique(group)
                # for g in unique_g:
                #     n_g = norm_rank_group[group == g]
                #     nr = n_g[:, 0].sort().indices
                #     norm_rank_group[group == g, 0] = nr.float()
                # loss_temp = loss_fct(norm_rank_group[:, 0], norm_rank_group[:, 1])
                # loss_norm_informativeness += loss_temp
                
                # use cosine similarity
                loss_temp = nn.CosineSimilarity(dim=-1)(n, rank)
                loss_norm_informativeness += loss_temp
        else:
            num_sent = pooler_output.size(1)
            if self.data_args.num_informative_pair == 2:
                n1 = pooler_output[:, range(0, num_sent, 2), :].norm(dim=-1)
                n2 = pooler_output[:, range(1, num_sent + 1, 2), :].norm(dim=-1)
                list_n = [n1, n2]
            elif self.data_args.num_informative_pair == 1:
                n = pooler_output[:, 1:, :].norm(dim=-1)
                list_n = [n]
            elif self.data_args.num_informative_pair == 0:
                n = pooler_output.norm(dim=-1)
                list_n = [n]
            else:
                raise NotImplementedError
                
            loss_norm_informativeness = 0
            for n in list_n:
                if self.model_args.loss_type == 'l1':
                    loss_norm_informativeness += F.l1_loss(n, n.sort()[0])
                elif self.model_args.loss_type == 'sl1':
                    loss_norm_informativeness += F.smooth_l1_loss(n, n.sort()[0])
                elif self.model_args.loss_type == 'mse':
                    loss_norm_informativeness += F.mse_loss(n, n.sort()[0])
                elif self.model_args.loss_type == 'margin':
                    assert n.size(1) == 2
                    n_intact, n_perturbed = n[:, 0], n[:, 1]
                    labels = torch.ones_like(n[:, 1])
                    loss_norm_informativeness += F.margin_ranking_loss(n_intact, n_perturbed, labels, margin=self.model_args.margin)
                else:
                    raise NotImplementedError

                    
        return loss_norm_informativeness
    
    def spearmans_rank_corr(self, nr, r):
        # spearman's rank correlation coefficient
        nr_mean = torch.mean(nr.float())
        r_mean = torch.mean(r)
        cov_nr_r = torch.mean((nr - nr_mean) * (r - r_mean))
        cov_nr_nr = torch.mean((nr - nr_mean) ** 2)
        cov_r_r = torch.mean((r - r_mean) ** 2)
        corr_nr_r = cov_nr_r / torch.sqrt(cov_nr_nr * cov_r_r)
        return corr_nr_r


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class CustomDropout(nn.Module):
    def __init__(self, dict_dropout):
        super(CustomDropout, self).__init__()
        self.dict_dropout = dict_dropout
        self.dropout = nn.Dropout(0.1)
        for i in range(len(dict_dropout)):
            setattr(self, f'dropout_{i}', nn.Dropout(dict_dropout[f'dropout_{i}']))
        
    def forward(self, x):
        out = x.clone()
        for i in range(len(self.dict_dropout)):
            dropout_i = getattr(self, f'dropout_{i}')
            # out[i::len(self.dict_dropout), :, :] = dropout_i(x[i::len(self.dict_dropout), :, :])
            # p = dropout_i.p #if i else 0
            out[i::len(self.dict_dropout), 1:, :] = dropout_i(x[i::len(self.dict_dropout), 1:, :]) #* (1-p) / 0.9
        out[:, 0, :] = self.dropout(out[:, 0, :])
        return out


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.infonormcorr = InformativenessNormCorrelation(cls.model_args, cls.data_args)
    cls.lambdas = cls.model_args.lambdas
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    step = 0,
    group=None,
    rank=None,
):
    # cls,encoder,position_ids,head_mask,inputs_embeds,labels,output_attentions,output_hidden_states,return_dict,mlm_input_ids,mlm_labels=model,model.bert,None,None,None,None,None,None,None,None,None
    # attention_mask, input_ids, token_type_ids = next(iter(train_dataloader)).values() # attention_mask, group, input_ids, rank, token_type_ids = next(iter(train_dataloader)).values()
    # input_ids, attention_mask, token_type_ids = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda() # input_ids, attention_mask, token_type_ids, rank, group = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), rank.cuda(), group.cuda()
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    # pooler_output = F.normalize(pooler_output,dim=2)
    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    # Hard negative
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)

    # z2 = z2.detach()

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))

    # Hard negative
    if num_sent >= 3:
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Calculate loss with hard negatives
    if num_sent == 3:
        # Note that weights are actually logits of weights
        z3_weight = cls.model_args.hard_negative_weight
        weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
        cos_sim = cos_sim + weights

    
    # contruct hard samples
    
    z3 = torch.cat([z2[1:],z2[:1]],dim=0)
    # lambdas = (1-F.cosine_similarity(z1,z3,dim=-1))*cls.lambdas/2
    # lambdas = lambdas.detach()
    # lambdas = [np.random.uniform(0.2,0.4) for _ in range(z3.size(0))]
    lambdas = [cls.lambdas] * z3.size(0)
    # lambdas = np.random.beta(5,3,size=z3.size(0))
    # z4 = (1-lambdas).unsqueeze(1) * z2 + lambdas.unsqueeze(1) * z3
    # cos_sim = cls.sim(z1.unsqueeze(1),z4.unsqueeze(0))
    lambdas = torch.Tensor(lambdas).to(cls.device)
    # z2_mix = (1 - lambdas).unsqueeze(1) * z2_mix
    z3 = lambdas.unsqueeze(1) * z2 + (1 - lambdas).unsqueeze(1) * z3
    z3 = z3.detach()
    z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
    mask = torch.eye(z1_z3_cos.size(0)).to(cls.device)
    mask = torch.cat([mask[-1:],mask[:-1]])
    mask = 1 - mask
    z1_z3_cos = mask * z1_z3_cos
    
    cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    z3_weight = cls.model_args.hard_negative_weight
    weights = torch.tensor(
        [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    ).to(cls.device)
    cos_sim = cos_sim + weights
    
    
    z4 = torch.cat([z1[1:],z1[:1]],dim=0)

    # lambdas = (1-F.cosine_similarity(z4,z2,dim=-1))*cls.lambdas/2
    # lambdas = lambdas.detach()
    # lambdas = [np.random.uniform(0.2,0.4) for _ in range(z3.size(0))]
    lambdas = [cls.lambdas] * z4.size(0)
    # lambdas = np.random.beta(5,3,size=z3.size(0))
    # z4 = (1-lambdas).unsqueeze(1) * z2 + lambdas.unsqueeze(1) * z3
    # cos_sim = cls.sim(z1.unsqueeze(1),z4.unsqueeze(0))
    lambdas = torch.Tensor(lambdas).to(cls.device)
    # z2_mix = (1 - lambdas).unsqueeze(1) * z2_mix
    z4 = lambdas.unsqueeze(1) * z1 + (1 - lambdas).unsqueeze(1) * z4
    z4 = z4.detach()
    z4_z2_cos = cls.sim(z4.unsqueeze(1), z2.unsqueeze(0))
    mask = torch.eye(z4_z2_cos.size(0)).to(cls.device)
    mask = torch.cat([mask[:,-1:],mask[:,:-1]],dim=1)
    # print('z4_z2_coas',z4_z2_cos)
    mask = 1 - mask
    z4_z2_cos = mask * z4_z2_cos
    
    cos_sim = torch.cat([cos_sim, z4_z2_cos], 1)
    z4_weight = cls.model_args.hard_negative_weight
    weights = torch.tensor(
        [[0.0] * (cos_sim.size(-1) - z4_z2_cos.size(-1)) + [0.0] * i + [z4_weight] + [0.0] * (z4_z2_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    ).to(cls.device)
    cos_sim = cos_sim + weights
    
    
    # F.pdist(z2).pow(2).mul(2).neg().exp().mean().log()) / 2
    loss = loss_fct(cos_sim, labels)

    # cos_sim_mix = cls.sim(z1.unsqueeze(1),z2_mix.unsqueeze(0))
    # loss_mix = loss_fct(cos_sim_mix,labels_mix)
    
    if cls.model_args.lambda_weight > 0:
        # self=InformativenessNormCorrelation(model_args, data_args)
        loss_informativeness = cls.infonormcorr(pooler_output, rank, group)
        loss += cls.model_args.lambda_weight * loss_informativeness
    
    # loss = loss + loss_mix
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.data_args = model_kargs["data_args"]
        self.bert = BertModel(config)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        step = 0,
        group=None,
        rank=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                step = step,
                group=group,
                rank=rank,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
