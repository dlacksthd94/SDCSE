import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from simcse.models import RobertaForCL, BertForCL, CustomDropout
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default='bert-based-uncased',
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls_before_pooler",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    
    # SDCSE's arguments
    lambda_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for lambda, which is the coefficient of informativeness-norm loss."
        }
    )
    loss_type: str = field(
        default='l1', 
        metadata={"help": "The type of loss function."}
    )
    margin: float = field(
        default=0.3, 
        metadata={"help": "The margin of margin loss. (only applied when margin loss is used)"}
    )
    
    def __post_init__(self):
        if self.loss_type is not None:
            assert self.loss_type in ['l1', 'sl1', 'mse', 'margin']

    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default='data/wiki1m_for_simcse.txt', 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    
    # SDCSE's arguments
    perturbation_type: str = field(
        default=None, 
        metadata={"help": "The type of perturbation applied to informative pairs"}
    )
    perturbation_num: int = field(
        default=2, 
        metadata={"help": "The number of perturbation applied to informative pairs"}
    )
    perturbation_step: int = field(
        default=1, 
        metadata={"help": "The step of perturbations applied to informative pairs"}
    )
    num_informative_pair: int = field(
        default=2, 
        metadata={
            "help": "The way to construct informative pairs. "
            "2: create two informative pairs, each of which is corresponded to each sentence example in a positive pair. "
            "1: create one informative pair just for first sentence example in a positive pair. "
            "0: use positive pairs as informative pairs."
        }
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
                
        if self.perturbation_type:
            assert self.perturbation_type in ['constituency_parsing', 'unk_token', 'mask_token', 'pad_token', 'dropout', 'none']
            if self.perturbation_type == 'none':
                self.perturbation_type = None
            else:
                if self.num_informative_pair is not None:
                    assert self.num_informative_pair in [2, 1, 0]
                    if self.num_informative_pair == 0:
                        assert self.perturbation_num == 1
            
        if self.perturbation_num > 0 or self.perturbation_step > 0:
            assert self.perturbation_num * self.perturbation_step <= self.max_seq_length - 2



@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

# #######################
# dict_plm = {
#     'bert_base': 'bert-base-uncased',
#     'bert_large': 'bert-large-uncased',
#     'roberta_base': 'roberta-base',
#     'roberta_large': 'roberta-large',
# }
# PLM='bert_base'
# BATCH_SIZE=128
# LR='3e-5'
# EPOCH=1
# SEED=0
# MAX_LEN=32
# LAMBDA='1e-0'
# PERTURB_TYPE='constituency_parsing'
# PERTURB_NUM=1
# PERTURB_STEP=1
# NUM_INFO_PAIR=0
# MARGIN='1e-1'
# sys.argv = [
#     'train.py',
#     '--model_name_or_path', dict_plm[PLM],
#     # '--model_name_or_path', f'result/my-unsup-simcse-{dict_plm[PLM]}_256_1e-4_1_0_32',
#     # '--train_file', 'data/wiki1m_for_simcse.txt',
#     '--train_file', '../data/backup_1000000/wiki1m_tree_cst_lg_large_subsentence.json',
#     '--output_dir', f'result/my-unsup-sdcse-{dict_plm[PLM]}_{BATCH_SIZE}_{LR}_{EPOCH}_{SEED}_{MAX_LEN}_{LAMBDA}',
#     '--num_train_epochs', str(EPOCH),
#     '--per_device_train_batch_size', str(BATCH_SIZE),
#     '--learning_rate', LR,
#     '--max_seq_length', str(MAX_LEN),
#     '--evaluation_strategy', 'steps',
#     '--metric_for_best_model', 'stsb_spearman',
#     '--load_best_model_at_end',
#     '--eval_steps', '125',
#     '--pooler_type', 'cls',
#     "--mlp_only_train",
#     '--overwrite_output_dir',
#     '--temp', '0.05',
#     '--do_train',
#     '--do_eval',
#     # '--eval_transfer',
#     '--fp16',
#     '--seed', str(SEED),
#     # '--no_cuda',
#     '--no_remove_unused_columns',
#     '--lambda_weight', str(LAMBDA),
#     '--perturbation_type', str(PERTURB_TYPE),
#     '--perturbation_num', str(PERTURB_NUM),
#     '--perturbation_step', str(PERTURB_STEP), 
#     '--num_informative_pair', str(NUM_INFO_PAIR),
#     '--margin', str(MARGIN),
# ]
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# #######################

# def main():
# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if (
    os.path.exists(training_args.output_dir)
    and os.listdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty."
        "Use --overwrite_output_dir to overcome."
    )

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info("Training/evaluation parameters %s", training_args)

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub
#
# For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
# behavior (see below)
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file
extension = data_args.train_file.split(".")[-1]
if extension == "txt":
    extension = "text"
if extension == "csv":
    datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
else:
    datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")

# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

if model_args.model_name_or_path:
    if 'roberta' in model_args.model_name_or_path:
        model = RobertaForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
            data_args=data_args
        )
    elif 'bert' in model_args.model_name_or_path:
        model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
            data_args=data_args
        )
        if model_args.do_mlm:
            pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
            model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
    else:
        raise NotImplementedError
else:
    raise NotImplementedError
    logger.info("Training new model from scratch")
    model = AutoModelForMaskedLM.from_config(config)

model.resize_token_embeddings(len(tokenizer))

if data_args.perturbation_type == 'dropout':
    if data_args.num_informative_pair == 2:
        dict_dropout = {f'dropout_{i}': round(0.1 + 0.1 * data_args.perturbation_step * i, 4) for i in range(data_args.perturbation_num + 1)}
    elif data_args.num_informative_pair == 1:
        dict_dropout = {f'dropout_{i}': round(0.1 + 0.1 * data_args.perturbation_step * (i - 1), 4) for i in range(1, data_args.perturbation_num + 2)}
        dict_dropout['dropout_0'] = 0.1
    elif data_args.num_informative_pair == 0:
        dict_dropout = {
            'dropout_0': 0.1,
            'dropout_1': 0.1 + 0.1 * data_args.perturbation_step
        }
    else:
        raise NotImplementedError
    dropout_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layer_names.append(name)
    dropout_layer_names

    # 드롭아웃 레이어 변경
    custom_dropout = CustomDropout(dict_dropout)
    plm_name = model_args.model_name_or_path.split('-')[0]
    for name in dropout_layer_names:
        plm = getattr(model, plm_name)
        if name.startswith(f'{plm_name}.embeddings'):
            plm.embeddings.dropout = custom_dropout
        elif name.startswith(f'{plm_name}.encoder'):
            n = int(name.split('.')[3])
            plm.encoder.layer[n].attention.self.dropout = custom_dropout
            plm.encoder.layer[n].attention.output.dropout = custom_dropout
            plm.encoder.layer[n].output.dropout = custom_dropout
    model
# elif data_args.perturbation_type in ['mask_token', 'unk_token']:
#     dict_token = {
#         'mask_token': '[MASK]',
#         'unk_token': '[UNK]',
#     }
#     token_id = tokenizer.convert_tokens_to_ids(dict_token[data_args.perturbation_type])
#     model.bert.embeddings.word_embeddings.weight.data[token_id] = torch.zeros(model.bert.embeddings.word_embeddings.embedding_dim)
#     # model.bert.embeddings.word_embeddings.weight.data[101] = torch.zeros(model.bert.embeddings.word_embeddings.embedding_dim) # make cls 0
#     model.bert.embeddings.word_embeddings.weight.data[102] = torch.zeros(model.bert.embeddings.word_embeddings.embedding_dim) # make sep 0
#     # model.bert.embeddings.word_embeddings.weight.data[103].norm() # unk 100, cls 101, sep 102, mask 103, pad 0

# Prepare features
column_names = datasets["train"].column_names
sent2_cname = None
if len(column_names) == 2:
    # Pair datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
elif len(column_names) == 3:
    if column_names[1] == 'group' and column_names[2] == 'rank':
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
elif len(column_names) == 1:
    # Unsupervised datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[0]
else:
    raise NotImplementedError

if data_args.perturbation_type == 'constituency_parsing' and data_args.num_informative_pair == 0:
    # datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")
    import pyarrow
    import pandas as pd
    from datasets.arrow_dataset import Dataset
    df = datasets.data['train'].to_pandas()
    df_groupby = df.groupby('group').agg(list)
    num_pair = data_args.perturbation_num + 1
    df_groupby['text'] = df_groupby['text'].apply(lambda x: x[:num_pair] if len(x) > 1 else x * num_pair)
    df_groupby['rank'] = df_groupby['rank'].apply(lambda x: x[:num_pair] if len(x) > 1 else x * num_pair)
    df_new = pd.DataFrame({'text': df_groupby.explode('text')['text'], 'group': df_groupby.explode('text')['text'].index, 'rank': df_groupby.explode('rank')['rank']})
    df_new = df_new.reset_index(drop=True)
    datasets['train'] = Dataset(pyarrow.lib.Table.from_pandas(df_new))

examples = datasets['train'][0:10]

def prepare_features(examples):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the 
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    total = len(examples[sent0_cname])

    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "
    
    if data_args.num_informative_pair == 2:
        if data_args.perturbation_type == 'constituency_parsing':
            raise NotImplementedError
        else:
            sentences = examples[sent0_cname] + examples[sent1_cname]
            sentences *= data_args.perturbation_num + 1
    elif data_args.num_informative_pair == 1:
        if data_args.perturbation_type == 'constituency_parsing':
            raise NotImplementedError
        else:
            sentences = examples[sent0_cname] + examples[sent1_cname] * (data_args.perturbation_num + 1)
    elif data_args.num_informative_pair == 0:
        if data_args.perturbation_type == 'constituency_parsing':
            sentences = examples[sent0_cname]
        else:
            sentences = examples[sent0_cname] + examples[sent1_cname]
    else:
        raise NotImplementedError
    
    # If hard negative exists
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    
    if data_args.perturbation_type == 'constituency_parsing':
        features = {}
        if sent2_cname is not None:
            raise NotImplementedError
        else:
            for key in sent_features:
                num_pair = data_args.perturbation_num + 1
                features[key] = [sent_features[key][i:i + num_pair] for i in range(0, total, num_pair)]
    else:
        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                # features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
                features[key] = [[sent_features[key][i + total * j] for j in range(len(sent_features[key]) // total)] for i in range(total)]
        
    if data_args.perturbation_type == 'constituency_parsing':
        if data_args.num_informative_pair == 0:
            if examples.get('group'):
                features['group'] = [[[examples['group'][i]], [examples['group'][i + 1]]] for i in range(0, len(examples['group']), 2)]
            if examples.get('rank'):
                features['rank'] = [[[examples['rank'][i]], [examples['rank'][i + 1]]] for i in range(0, len(examples['rank']), 2)]
        elif data_args.num_informative_pair == 1:
            raise NotImplementedError
        elif data_args.num_informative_pair == 2:
            if examples.get('group'):
                features['group'] = [[[x]] * 2 for x in examples['group']]
            if examples.get('rank'):
                features['rank'] = [[[x]] * 2 for x in examples['rank']]
    
    return features
    prepare_features(examples)

if training_args.do_train:
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
next(iter(train_dataset))
from torch.utils.data import Subset
features = Subset(train_dataset, range(10))
train_dataset[:10]

# Data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = data_args.mlm_probability

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels', 'group', 'rank']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if model_args.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
        
        if data_args.perturbation_type not in [None, 'dropout', 'constituency_parsing']:
            # inputs=batch["input_ids"]
            self.perturbation(inputs=batch["input_ids"])
            # inputs[:10][:10]

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def perturbation(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        if data_args.perturbation_type == 'unk_token':
            perturbation_token_id = self.tokenizer.unk_token_id
        elif data_args.perturbation_type == 'mask_token':
            perturbation_token_id = self.tokenizer.mask_token_id
        elif data_args.perturbation_type == 'pad_token':
            perturbation_token_id = self.tokenizer.pad_token_id
        else:
            raise NotImplementedError

        size_sentence_pair = (data_args.perturbation_num + 1) * 2 if data_args.num_informative_pair else 2
        
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs[::size_sentence_pair].tolist()
        ]
        special_tokens_mask = (~torch.tensor(special_tokens_mask, dtype=torch.bool)).float()
        total_perturbation_num = data_args.perturbation_num * data_args.perturbation_step
        condition = special_tokens_mask.sum(dim=-1) == 0
        special_tokens_mask = torch.where(condition[:, None], torch.cat([torch.ones((special_tokens_mask.shape[0], 1)), special_tokens_mask[:, 1:]], dim=1), special_tokens_mask) # a subset of `special_tokens_mask` where all elements are 0 raises an error, so fill their 2nd index with 1.
        list_randint = torch.multinomial(special_tokens_mask, total_perturbation_num, replacement=False)
        
        if data_args.num_informative_pair == 2:
            for i in range(data_args.perturbation_num):
                for j in range((i + 1) * data_args.perturbation_step):
                    target_index = (range(list_randint[:, j].shape[0]), list_randint[:, j])
                    inputs[(i + 1) * 2::size_sentence_pair][target_index] = (torch.Tensor([perturbation_token_id] * target_index[1].shape[0]) * special_tokens_mask[target_index]).long()
                    inputs[(i + 1) * 2 + 1::size_sentence_pair][target_index] = (torch.Tensor([perturbation_token_id] * target_index[1].shape[0]) * special_tokens_mask[target_index]).long()
        elif data_args.num_informative_pair == 1:
            for i in range(data_args.perturbation_num):
                for j in range((i + 1) * data_args.perturbation_step):
                    target_index = (range(list_randint[:, j].shape[0]), list_randint[:, j])
                    inputs[i + 2::size_sentence_pair][target_index] = (torch.Tensor([perturbation_token_id] * target_index[1].shape[0]) * special_tokens_mask[target_index]).long()
        elif data_args.num_informative_pair == 0:
            # for j in range(data_args.perturbation_step):
            #     target_index = (range(list_randint[:, j].shape[0]), list_randint[:, j])
            #     inputs[1::size_sentence_pair][target_index] = (torch.Tensor([perturbation_token_id] * target_index[1].shape[0]) * special_tokens_mask[target_index]).long()
            probability_matrix = torch.full(inputs[1::2].shape, 0.1 * data_args.perturbation_step)
            probability_matrix.masked_fill_(~special_tokens_mask.bool(), value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            inputs[1::2][masked_indices] = perturbation_token_id
        else:
            raise NotImplementedError
        return

data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

trainer = CLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.model_args = model_args
trainer.data_args = data_args

# Training
if training_args.do_train:
    model_path = (
        model_args.model_name_or_path
        if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        else None
    )
    
    # self,trial=trainer,None
    train_result = trainer.train(model_path=model_path)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

# Evaluation
results = {}
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    results = trainer.evaluate(eval_senteval_transfer=True)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

# return results

# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()


# if __name__ == "__main__":
#     main()