import os
import re
import torch
import sys
import types
from typing import List, Optional, Union

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import (
    ctakes_tok,
    model_dicts,
    get_sentences_and_labels,
    assemble,
    get_model_pairs,
    get_eval_predictions,
)

from .pipelines.tagging import TaggingPipeline

from .cnlp_processors import (
    cnlp_processors,
    cnlp_output_modes,
    tagging,
    classification
)

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser

from itertools import chain, groupby

SPECIAL_TOKENS = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']

modes = ["inf", "eval"]


@dataclass
class PipelineArguments:
    """
    Arguments pertaining to the models, mode, and data used by the pipeline.
    """
    models_dir: str = field(
        metadata={
            "help": (
                "Path where each entity model is stored"
                "in a folder named after its corresponding cnlp_processor,"
                "models with a 'tagging' output mode will be run first"
                "followed by models with a 'classification'"
                "ouput mode over the assembled data"
            )
        }
    )
    in_file: str = field(
        metadata={
            "help": (
                "Path to file, with one raw sentence"
                "per line in the case of inference,"
                " and one <label>\t<annotated sentence> "
                "per line in the case of evaluation"
            )
        }
    )
    mode: str = field(
        default="inf",
        metadata={
            "help": (
                "Use mode for full pipeline, "
                "inference, which outputs annotated sentences"
                "and their relation, or eval, "
                "which outputs metrics for a provided set of samples "
                "(requires labels)"
            )
        }
    )
    tokenizer: str = field(
        default="roberta-base",
        metadata={
            "help": (
                "NEEDS TO BE DELETED (each tokenizer in model folder)"
            )
        }
    )
    axis_task: str = field(
        default="dphe_med",
        metadata={
            "help": (
                "key of the task in cnlp_processors "
                "which generates the tag that will map to <a1> <mention> </a1>"
                " in pairwise annotations"
            )
        }
    )


def main():
    parser = HfArgumentParser(PipelineArguments)

    if (
            len(sys.argv) == 2
            and sys.argv[1].endswith(".json")
    ):
        # If we pass only one argument to the script
        # and it's the path to a json file,
        # let's parse it to get our arguments.

        # the ',' is to deal with unary tuple weirdness
        pipeline_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        pipeline_args, = parser.parse_args_into_dataclasses()

    if pipeline_args.mode == "inf":
        inference(pipeline_args)
    elif pipeline_args.mode == "eval":
        evaluation(pipeline_args)
    else:
        ValueError("Invalid pipe mode!")


def inference(pipeline_args):
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    tokenizer = AutoTokenizer.from_pretrained(
        pipeline_args.tokenizer,
        # cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        additional_special_tokens=SPECIAL_TOKENS,
    )

    taggers_dict, out_model_dict = model_dicts(
        pipeline_args.models_dir,
        tokenizer
    )

    sent_processor = None

    for key in out_model_dict.keys():
        sent_processor = key

    _, sentences = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="inf",
        task_processor=cnlp_processors[sent_processor](),
    )

    annotated_sents = assemble(
        sentences,
        taggers_dict,
        pipeline_args.axis_task,
    )

    softmax = torch.nn.Softmax(dim=1)

    for ann_sent in annotated_sents:
        ann_encoding = tokenizer(
            ctakes_tok(ann_sent),
            max_length=128,   # UN-HARDCODE
            return_tensors='pt',
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        for out_task, model in out_model_dict.items():
            model.eval() # wew lad
            out_task_processor = cnlp_processors[out_task]()
            out_task_labels = out_task_processor.get_labels()
            with torch.no_grad():
                model_outputs = model(**ann_encoding)
            logits = model_outputs["logits"][0]
            scores = softmax(logits)
            out_label_idx = torch.argmax(scores).item()
            label = out_task_labels[out_label_idx]
            print(f"{label} : {ann_sent}")


def evaluation(pipeline_args):
    # holding off on this for now

    # used to get the relevant models from the label name,
    # e.g. med-dosage -> med, dosage -> dphe-med, dphe-dosage,
    # -> taggers_dict[dphe-med], taggers_dict[dphe-dosage]

    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    tokenizer = AutoTokenizer.from_pretrained(
        pipeline_args.tokenizer,
        # cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        additional_special_tokens=SPECIAL_TOKENS,
    )

    taggers_dict, out_model_dict = model_dicts(
        pipeline_args.models_dir,
        tokenizer
    )

    sent_processor = None

    for key in out_model_dict.keys():
        sent_processor = key

    out_task_processor=cnlp_processors[sent_processor]()
    
    
    idx_labels, str_labels, annotated_sents = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="eval",
        task_processor=out_task_processor,
    )

    
    
    model_pairs = get_model_pairs(str_labels, taggers_dict)
    deannotated_sents = list(map(lambda s : re.sub(r"</?a[1-2]>", "", s), annotated_sents))
    # strip the sentence of tags
    
    predictions = get_eval_predictions(
        model_pairs,
        deannotated_sents,
        taggers_dict,
        out_model_dict,
        tokenizer,
        pipeline_args.axis_task,
    )
    return 0


if __name__ == "__main__":
    main()
