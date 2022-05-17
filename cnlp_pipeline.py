import os
import re
import sys
import numpy as np

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import (
    model_dicts,
    get_sentences_and_labels,
    assemble,
    get_model_pairs,
    get_eval_predictions,
)

from .cnlp_processors import cnlp_compute_metrics

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import AutoConfig, AutoModel, HfArgumentParser

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
                "in a folder named after its "
                "corresponding cnlp_processor, "
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
        mode='inf',
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
    ann_sent_groups = assemble(
        sentences,
        taggers_dict,
        pipeline_args.axis_task,
    )

    for out_task, out_pipe in out_model_dict.items():
        # Get the output for each relation classifier models,
        # tokenizer_kwargs are passed directly
        # text classification pipelines during __call__
        # (Huggingface's idea not mine)
        for ann_sent_group in ann_sent_groups:
            axis_mention_dict = {}
            for sent_dict in ann_sent_group:
                main_offsets = sent_dict['main_offsets']
                ann_sent = sent_dict['sentence']
                pipe_output = out_pipe(
                    ann_sent,
                    padding="max_length",
                    truncation=True,
                    is_split_into_words=True,
                )
                
                strongest_label = max(pipe_output[0], key=lambda d: d['score'])

                # print(f"{main_offsets} : {ann_sent}")
                # print(f"{pipe_output[0]}")
                # print(f"{strongest_label}")
                
                def label_update(label_dict, mention_dict, offsets):
                    new_label = label_dict['label']
                    new_score = label_dict['score']
                    no_label = new_label not in mention_dict[main_offsets].keys()
                    if no_label:
                        return no_label
                    else:
                        # higher score
                        return new_score > mention_dict[offsets][new_label]['score']
                        

                if main_offsets not in axis_mention_dict.keys():
                    axis_mention_dict[main_offsets] = {}
                    axis_label_dict = {
                        'sentence' : ann_sent,
                        'score' : strongest_label['score'],
                    }
                    axis_mention_dict[main_offsets][strongest_label['label']] = axis_label_dict
                elif label_update(strongest_label, axis_mention_dict, main_offsets):
                    axis_label_dict = {
                        'sentence' : ann_sent,
                        'score' : strongest_label['score'],
                    }
                    axis_mention_dict[main_offsets][strongest_label['label']] = axis_label_dict
            for labeled_dict in axis_mention_dict.values():
                for label, sent in labeled_dict.items():
                    print(f"{label}, {sent['score']} : {sent['sentence']}")
            

def evaluation(pipeline_args):
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(
        pipeline_args.models_dir,
        mode='eval',
    )

    # Assume sentences are annotated and clean them anyway
    # for eval we will need the index based labels organized by out task
    # as well as the string labels organized by out task
    # (for finding relevant model pairs)
    (
        idx_labels_dict,
        str_labels_dict,
        annotated_sents
    ) = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="eval",
        task_names=out_model_dict.keys(),
    )

    # Get the model (by task_name) pairs for each instance e.g.
    # med-dosage -> (dphe_med, dphe_dosage)
    model_pairs_dict = get_model_pairs(str_labels_dict, taggers_dict)

    # Remove annotations from the sentence e.g.
    # '<a1> tamoxifen </a1>, <a2> 20 mg </a2> once daily'
    # -> 'tamoxifen , 20 mg once daily'
    def deannotate(s):
        return re.sub(r"</?a[1-2]>", "", s)
    
    deannotated_sents = map(deannotate, annotated_sents)

    # Get dictionary of predictions
    # indexed by task
    predictions_dict = get_eval_predictions(
        model_pairs_dict,
        deannotated_sents,
        taggers_dict,
        out_model_dict,
        pipeline_args.axis_task,
    )

    # Print the metrics by task
    for task_name, predictions in predictions_dict.items():
        report = cnlp_compute_metrics(
            task_name,
            np.array(predictions),
            np.array(idx_labels_dict[task_name])
        )
        print(report)


if __name__ == "__main__":
    main()
