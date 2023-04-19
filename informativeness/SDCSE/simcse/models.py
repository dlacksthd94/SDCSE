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
    
    def __init__(self, loss_type):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_type = loss_type
    
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
            n1 = pooler_output[:, range(0, num_sent, 2), :].norm(dim=-1)
            # n2 = pooler_output[:, range(1, num_sent + 1, 2), :].norm(dim=-1)
            
            if self.loss_type == 'l1':
                loss_norm_informativeness = F.l1_loss(n1, n1.sort()[0])
            elif self.loss_type == 'sl1':
                loss_norm_informativeness = F.smooth_l1_loss(n1, n1.sort()[0])
            elif self.loss_type == 'mse':
                loss_norm_informativeness = F.mse_loss(n1, n1.sort()[0])
                
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
            first_hidden = hidden_states[1]
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
    def __init__(self, dict_dropout, perturbation_num):
        super(CustomDropout, self).__init__()
        self.perturbation_num = perturbation_num
        for i in range(perturbation_num + 1):
            setattr(self, f'dropout_{i}', nn.Dropout(dict_dropout[f'dropout_{i}']))
        
    def forward(self, x):
        out = torch.tensor(x)
        for i in range(self.perturbation_num + 1):
            out[i::self.perturbation_num + 1, 1:, :] = getattr(self, f'dropout_{i}')(x[i::self.perturbation_num + 1, 1:, :])
        return out


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.infonormcorr = InformativenessNormCorrelation(cls.model_args.loss_type)
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

    # Separate representation
    # z1, z2 = pooler_output[:,0], pooler_output[:,1]
    dict_z = {}
    for i in range(num_sent):
        dict_z[f'z{i + 1}'] = pooler_output[:, i] 
    
    if group is not None and rank is not None:
        dict_z['z1'] = dict_z['z1'][rank[:, 0].squeeze() == 0]
        dict_z['z2'] = dict_z['z2'][rank[:, 0].squeeze() == 0]
        
    # # Hard negative
    # if num_sent == 3:
    #     z3 = pooler_output[:, 2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # Gather hard negative
        # if num_sent >= 3:
        #     z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        #     dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
        #     z3_list[dist.get_rank()] = z3
        #     z3 = torch.cat(z3_list, 0)

        # Dummy vectors for allgather
        dict_z_list = {}
        for i in range(num_sent):
            dict_z_list[f'z{i + 1}_list'] = [torch.zeros_like(dict_z[f'z{i + 1}']) for _ in range(dist.get_world_size())]
        
        # Allgather
        for i in range(num_sent):
            dist.all_gather(tensor_list=dict_z_list[f'z{i + 1}_list'], tensor=dict_z[f'z{i + 1}'].contiguous())
        
        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        for i in range(num_sent):
            dict_z_list[f'z{i + 1}_list'][dist.get_rank()] = dict_z[f'z{i + 1}']

        # Get full batch embeddings: (bs x N, hidden)
        for i in range(num_sent):
            dict_z[f'z{i + 1}'] = torch.cat(dict_z_list[f'z{i + 1}_list'], 0)
        
    cos_sim = cls.sim(dict_z['z1'].unsqueeze(1), dict_z['z2'].unsqueeze(0))
    # # Hard negative
    # if num_sent >= 3:
    #     z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
    #     cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # # Calculate loss with hard negatives
    # if num_sent == 3:
    #     # Note that weights are actually logits of weights
    #     z3_weight = cls.model_args.hard_negative_weight
    #     weights = torch.tensor(
    #         [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
    #     ).to(cls.device)
    #     cos_sim = cos_sim + weights

    # self,input,target=loss_fct,cos_sim,labels 
    loss = loss_fct(cos_sim, labels)# + loss_fct(cos_sim_2, labels)
    
    if cls.model_args.lambda_weight > 0:
        loss_informativeness = cls.infonormcorr(pooler_output, rank, group)
        loss += cls.model_args.lambda_weight * loss_informativeness
            
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
        logits=(cos_sim),
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
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
        self.bert = BertModel(config, add_pooling_layer=False)

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
                group=group,
                rank=rank,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

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
