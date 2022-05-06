import os
import re
import torch
import sys
import types
import json
import numpy as np
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
    classification,
    cnlp_compute_metrics,
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
                "Path where each entity model is stored "
                "in a folder named after its corresponding cnlp_processor, "
                "models with a 'tagging' output mode will be run first "
                "followed by models with a 'classification' "
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
                "inference, which outputs annotated sentences "
                "and their relation, or eval, "
                "which outputs metrics for a provided set of samples "
                "(requires labels)"
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
    # Required for loading cnlpt models using Huggingface
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(
        pipeline_args.models_dir,
    )

    # Only need raw sentences for inference
    _, _, sentences = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="inf",
        task_names=out_model_dict.keys(),
    )

    # Annotate the sentences with entities
    # using the taggers. e.g.
    # 'tamoxifen , 20 mg once daily' 
    # (dphe_strength) -> '<a1> tamoxifen </a1>, <a2> 20 mg </a2> once daily' 
    # (dphe_freq)-> '<a1> tamoxifen </a1>, 20 mg <a2> once daily </a2>'
    # etc.
    annotated_sents = assemble(
        sentences,
        taggers_dict,
        pipeline_args.axis_task,
    )

    for out_task, out_pipe in out_model_dict.items():
        # Get the output for each relation classifier models,
        # tokenizer_kwargs are passed directly
        # text classification pipelines during __call__
        # (Huggingface's idea not mine)
        pipe_output = out_pipe(
            annotated_sents,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        # For now just print the system annotated sentence
        # plus its predicted relation label
        for out, sent in zip(pipe_output, annotated_sents):
            print(f"{out['label']} : {sent}")


def evaluation(pipeline_args):
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(
        pipeline_args.models_dir
    )

    # Assume sentences are annotated and clean them anyway
    # for eval we will need the index based labels organized by out task
    # as well as the string labels organized by out task (for finding relevant model pairs)
    idx_labels_dict, str_labels_dict, annotated_sents = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="eval",
        task_names=out_model_dict.keys(),
    ) 

    # Get the model (by task_name) pairs for each instance e.g.
    # med-dosage -> (dphe_med, dphe_dosage)
    model_pairs_dict = get_model_pairs(str_labels_dict, taggers_dict)

    # Remove annotations from the sentence e.g.
    #'<a1> tamoxifen </a1>, <a2> 20 mg </a2> once daily'
    # -> 'tamoxifen , 20 mg once daily'
    deannotated_sents = list(map(lambda s : re.sub(r"</?a[1-2]>", "", s), annotated_sents))
    
    predictions_dict = get_eval_predictions(
        model_pairs_dict,
        deannotated_sents,
        taggers_dict,
        out_model_dict,
        pipeline_args.axis_task,
    )

    for task_name, predictions in predictions_dict.items(): 
        report = cnlp_compute_metrics(task_name, np.array(predictions), np.array(idx_labels_dict[task_name]))
        print(report)


if __name__ == "__main__":
    main()
