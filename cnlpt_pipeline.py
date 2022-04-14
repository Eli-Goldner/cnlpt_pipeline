import dataclasses
import sys
import os

from typing import Callable, Dict, Optional, List, Union

from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, AutoModel, EvalPrediction, HfArgumentParser
from .cnlp_processors import cnlp_processors, cnlp_output_modes, cnlp_compute_metrics, tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset, DataTrainingArguments

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

cnlpt_models = ['cnn', 'lstm', 'hier', 'cnlpt']

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model: Optional[str] = field( default='cnlpt', 
        metadata={'help': "Model type", 'choices':cnlpt_models}
    )
    encoder_name: Optional[str] = field(default='roberta-base',
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    layer: Optional[int] = field(
        default=-1, metadata={"help": "Which layer's CLS ('<s>') token to use"}
    )
    token: bool = field(
        default=False, metadata={"help": "Classify over an actual token rather than the [CLS] ('<s>') token -- requires that the tokens to be classified are surrounded by <e>/</e> tokens"}
    )
    num_rel_feats: Optional[int] = field(
        default=12, metadata={"help": "Number of features/attention heads to use in the NxN relation classifier"}
    )
    head_features: Optional[int] = field(
        default=64, metadata={"help": "Number of parameters in each attention head in the NxN relation classifier"}
    )
    use_prior_tasks: bool = field(
        default=False, metadata={"help": "In the multi-task setting, incorporate the logits from the previous tasks into subsequent representation layers. This will be done in the task order specified in the command line."}
    )
    hier_num_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "For the hierarchical model, the number of document-level transformer "
                "layers"
            )
        },
    )
    hier_hidden_dim: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "For the hierarchical model, the inner hidden size of the positionwise "
                "FFN in the document-level transformer layers"
            )
        },
    )
    hier_n_head: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the number of attention heads in the "
                "document-level transformer layers"
            )
        },
    )
    hier_d_k: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the size of the query and key vectors in "
                "the document-level transformer layers"
            )
        },
    )
    hier_d_v: Optional[int] = field(
        default=96,
        metadata={
            "help": (
                "For the hierarchical model, the size of the value vectors in the "
                "document-level transformer layers"
            )
        },
    )



    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

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
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    assert len(data_args.task_name) == len(data_args.data_dir), 'Number of tasks and data directories should be the same!'

    # Note Tim's trick of having the tokenizer used to train the model being required
    # (if the encoder name is a path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name, # if model_args.tokenizer_name else model_args.encoder_name,
        cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        additional_special_tokens=['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
    )
    
    model_name = model_args.model
    hierarchical = model_name == 'hier'
    
    # Get datasets
    train_dataset = (
        ClinicalNlpDataset(
            data_args,
            tokenizer=tokenizer,
            cache_dir=model_args.cache_dir,
            hierarchical=hierarchical
        ) if training_args.do_train else None
    )
    eval_dataset = (
        ClinicalNlpDataset(
            data_args,
            tokenizer=tokenizer,
            mode="dev",
            cache_dir=model_args.cache_dir,
            hierarchical=hierarchical
        )
        if training_args.do_eval else None
    )
    test_dataset = (
        ClinicalNlpDataset(
            data_args,
            tokenizer=tokenizer,
            mode="test",
            cache_dir=model_args.cache_dir,
            hierarchical=hierarchical
        )
        if training_args.do_predict else None
    )
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.encoder_name,
        cache_dir=model_args.cache_dir,
    )
    
    # setting 2) evaluate or make predictions
    model = CnlpModelForClassification.from_pretrained(
        model_args.encoder_name,
        config=config,
        class_weights=None if train_dataset is None else train_dataset.class_weights,
        final_task_weight=training_args.final_task_weight,
        freeze=training_args.freeze,
        bias_fit=training_args.bias_fit,
        argument_regularization=training_args.arg_reg
    )
    
    
if __name__ == "__main__":
    main()
