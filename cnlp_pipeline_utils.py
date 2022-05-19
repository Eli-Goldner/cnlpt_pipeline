import os
import re
import warnings

from .cnlp_processors import (
    cnlp_processors,
    cnlp_output_modes,
    tagging,
    classification
)

from .pipelines.tagging import TaggingPipeline
from .pipelines.classification import ClassificationPipeline
from .pipelines import ctakes_tok

from .CnlpModelForClassification import CnlpModelForClassification

from transformers import AutoConfig, AutoTokenizer

from heapq import merge

from itertools import chain, groupby, tee, zip_longest

SPECIAL_TOKENS = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']


# Get dictionary of entity tagging models/pipelines
# and relation extraction models/pipelines
# both indexed by task names
def model_dicts(models_dir, mode='inf'):
    taggers_dict = {}
    out_model_dict = {}

    # For each folder in the model_dir...
    for file in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, file)
        task_name = str(file)
        if os.path.isdir(model_dir) and task_name in cnlp_processors.keys():

            # Load the model, model config, and model tokenizer
            # from the model foldner
            config = AutoConfig.from_pretrained(
                model_dir,
            )

            model = CnlpModelForClassification.from_pretrained(
                model_dir,
                config=config,
            )

            # Right now assume roberta thus
            # add_prefix_space = True
            # but want to generalize eventually to
            # other model tokenizers, in particular
            # Flair models for RadOnc
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                add_prefix_space=True,
                additional_special_tokens=SPECIAL_TOKENS,
            )

            task_processor = cnlp_processors[task_name]()

            # Add tagging pipelines to the tagging dictionary
            if cnlp_output_modes[task_name] == tagging:
                taggers_dict[task_name] = TaggingPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor
                )
            # Add classification pipelines to the classification dictionary
            elif cnlp_output_modes[task_name] == classification:
                classifier = None
                if mode == 'inf':
                    classifier = ClassificationPipeline(
                        model=model,
                        tokenizer=tokenizer,
                        return_all_scores=True,
                        task_processor=task_processor
                    )
                elif mode == 'eval':
                    classifier = ClassificationPipeline(
                        model=model,
                        tokenizer=tokenizer,
                        task_processor=task_processor
                    )
                else:
                    ValueError("Invalid processing mode")
                out_model_dict[task_name] = classifier
            # Tasks other than tagging and sentence/relation classification
            # not supported for now since I wasn't sure how to fit them in
            else:
                ValueError(
                    (
                        "output mode "
                        f"{cnlp_output_modes[task_name]}"
                        "not currently supported"
                    )
                )
    return taggers_dict, out_model_dict


# Given the tagger pipelines, axial/central/anchor task
# (which produces the axial entity for the relations)
# and raw sentences, this function produces the
# list of sentences annotated with predictions
# from the tagging model
# - this master function is for inference mode only,
# since for evaluation the taggers used vary sentence by sentence
def assemble(sentences, taggers_dict, axis_task):
    # Return one list of all the system prediction annotations
    # for each sentences
    def process(sent):
        return _assemble(sent, taggers_dict, axis_task, mode='inf')
    return map(process, sentences)


# Given a dictionary of taggers, a single sentence, and an axial task,
# return the sentence annotated by the taggers
# - can be used for inference mode by passing the entire dictionary of taggers
# - can be used for eval mode by passing the model pair for that sentence
def _assemble(sentence, taggers_dict, axis_task, mode='inf'):
    axis_pipe = taggers_dict[axis_task]
    axis_ann = axis_pipe(sentence)
    # We split by mode cases since there are optmizations
    # helpful for inference that could cause problems for
    # evaluation
    if mode == 'inf':
        ann_sents = []
        # Make sure there's at least one axial mention
        # before running any other taggers
        if any(filter(lambda x: x[0] == 'B', axis_ann)):
            sig_ann_ls = []
            for task, pipeline in taggers_dict.items():
                # Don't rerun the axial pipeline
                if task != axis_task:
                    sig_out = pipeline(sentence)
                    sig_ann_ls.append(sig_out)
            ann_sents = merge_annotations(
                axis_ann,
                sig_ann_ls,
                sentence,
                mode='inf'
            )
    elif mode == 'eval':
        # For eval, we need to run both taggers regardless of
        # whether they miss
        ann_sents = []
        sig_ann_ls = []
        for task, pipeline in taggers_dict.items():
            if task != axis_task:
                sig_out = pipeline(sentence)
                sig_ann_ls.append(sig_out)
        ann_sents = merge_annotations(
            axis_ann,
            sig_ann_ls,
            sentence,
            mode='eval',
        )
    else:
        ValueError("Invalid processing mode: {model}")
    return ann_sents


# Convert from B I O list format
# to 120 string format for easy
# pattern matching
def get_partitions(annotation):
    def tag2idx(tag):
        if tag != 'O':
            if tag[0] == 'B':
                return '1'
            elif tag[0] == 'I':
                return '2'
        # 2 identifies the second
        else:
            return '0'
    return ''.join(map(tag2idx, annotation))


def process_ann(annotation):
    span_begin, span_end = 0, 0
    indices = []
    partitions = get_partitions(annotation)
    # Group 1's individually as well as 1's followed by
    # any nummber of 2's, e.g.
    # 00000011111112222121212
    # -> 000000 1 1 1 1 1 1 12222 12 12 12
    for span in filter(None, re.split(r'(12*)', partitions)):
        span_end = len(span) + span_begin - 1
        if span[0] == '1':
            # Get indices in list/string of each span
            # which describes a mention
            indices.append((span_begin, span_end))
        span_begin = span_end + 1
    return indices


# Given a raw sentence, an axial entity tagging,
# and a non-axial entity tagging,
# return the sentence with XML tags
# around the entities
# - this is another wrapper function where
# we handle looping and some optimizations
def merge_annotations(axis_ann, sig_ann_ls, sentence, mode='inf'):
    merged_annotations = []
    axis_indices = process_ann(axis_ann)
    sig_indices_ls = [process_ann(sig_ann) for sig_ann in sig_ann_ls]
    ref_sent = ctakes_tok(sentence)
    if mode == 'eval':
        for sig_indices in sig_indices_ls:
            ann_sent = ref_sent.copy()
            intersects = get_intersect(sig_indices, axis_indices)
            if intersects:
                warnings.warn(
                    (
                        "Warning axis annotation and sig annotation \n"
                        f"{ref_sent}\n"
                        f"{axis_indices}\n"
                        f"{sig_indices}\n"
                        f"Have intersections at indices:\n"
                        f"{intersects}"
                    )
                )
            else:
                for axis, sig in zip_longest(axis_indices, sig_indices):
                    if sig is not None:
                        s1, s2 = sig
                        ann_sent[s1] = '<a2> ' + ann_sent[s1]
                        ann_sent[s2] = ann_sent[s2] + ' </a2>'
                    if axis is not None:
                        a1, a2 = axis
                        ann_sent[a1] = '<a1> ' + ann_sent[a1]
                        ann_sent[a2] = ann_sent[a2] + ' </a1>'
                merged_annotations.append(' '.join(ann_sent))
    elif mode == 'inf':
        for a1, a2 in axis_indices:
            # list of lists, only want to
            # iterate if at least one is non-empty
            if any(sig_indices_ls):
                for sig_indices in sig_indices_ls:
                    for s1, s2 in sig_indices:
                        intersects = get_intersect([(a1, a2)], [(s1, s2)])
                        if intersects:
                            warnings.warn(
                                (
                                    "Warning axis annotation and sig annotation \n"
                                    f"{ref_sent}\n"
                                    f"{a1, a2}\n"
                                    f"{s1, s2}\n"
                                    f"Have intersections at indices:\n"
                                    f"{intersects}"
                                )
                            )
                        ann_sent = ref_sent.copy()
                        ann_sent[a1] = '<a1> ' + ann_sent[a1]
                        ann_sent[a2] = ann_sent[a2] + ' </a1>'
                        ann_sent[s1] = '<a2> ' + ann_sent[s1]
                        ann_sent[s2] = ann_sent[s2] + ' </a2>'
                        
                        sent_dict = {'sentence' : ' '.join(ann_sent), 'main_offsets' : (a1, a2)}
                        merged_annotations.append(sent_dict)
            else:
                ann_sent = ref_sent.copy()
                ann_sent[a1] = '<a1> ' + ann_sent[a1]
                ann_sent[a2] = ann_sent[a2] + ' </a1>'
                sent_dict = {'sentence' : ' '.join(ann_sent), 'main_offsets' : (a1, a2)}
                merged_annotations.append(sent_dict)
    else:
        ValueError(f"Invalid processsing mode : {mode}")
    return merged_annotations


# Shamelessly grabbed (and adapted) from https://stackoverflow.com/a/57293089
def get_intersect(ls1, ls2):
    m1, m2 = tee(merge(ls1, ls2, key=lambda k: k[0]))
    next(m2, None)
    out = []
    for v, g in groupby(zip(m1, m2), lambda k: k[0][1] < k[1][0]):
        if not v:
            ls = [*g][0]
            inf = max(i[0] for i in ls)
            sup = min(i[1] for i in ls)
    return out


# Get dictionary of final pipeline predictions
# over annotated sentences, indexed by task name
def get_eval_predictions(
        model_pairs_dict,
        deannotated_sents,
        taggers_dict,
        out_model_dict,
        axis_task,
):
    reannotated_sents = {}
    for task_name, task_model_pairs in model_pairs_dict.items():
        reannotated_sents[task_name] = []
        for sent, pair in zip(deannotated_sents, task_model_pairs):
            tagger_pair_dict = {key: taggers_dict[key] for key in pair}
            reann_sent_ls = _assemble(
                sent,
                tagger_pair_dict,
                axis_task,
                mode='eval'
            )
            assert len(reann_sent_ls) == 1, (
                "_assemble is misbehaving: \n"
                f"{reann_sent_ls}"
            )
            reannotated_sents[task_name].append(reann_sent_ls[0])

    out_labels = None
    predictions_dict = {}

    for out_task, out_pipe in out_model_dict.items():

        out_task_processor = cnlp_processors[out_task]()
        out_task_labels = out_task_processor.get_labels()

        out_label_map = {label: i for i, label in enumerate(out_task_labels)}
        out_labels = []
        for reann_sent in reannotated_sents[out_task]:
            pipe_output = out_pipe(
                reann_sent,
                # Again, classification pipelines
                # take tokenizer args during __call__
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
            )
            out_labels.append(out_label_map[pipe_output[0]['label']])
        predictions_dict[out_task] = out_labels

        return predictions_dict


# Get raw sentences, as well as
# dictionaries of labels indexed by output pipeline
# task names
def get_sentences_and_labels(in_file: str, mode: str, task_names):
    task_processors = [cnlp_processors[task_name]() for task_name
                       in task_names]
    idx_labels_dict, str_labels_dict = {}, {}
    if mode == "inf":
        # 'test' let's us forget labels
        # just use the first task processor since
        # _create_examples and _read_tsv are task/label agnostic
        examples = task_processors[0]._create_examples(
            task_processors[0]._read_tsv(in_file),
            "test"
        )
    elif mode == "eval":
        # 'dev' lets us get labels without running into issues of downsampling
        examples = task_processors[0]._create_examples(
            task_processors[0]._read_tsv(in_file),
            "dev"
        )

        def example2label(example):
            if isinstance(example.label, list):
                return [label_map[label] for label in example.label]
            else:
                return label_map[example.label]

        for task_name, task_processor in zip(task_names, task_processors):
            label_list = task_processor.get_labels()
            label_map = {label: i for i, label in enumerate(label_list)}

            if examples[0].label:
                idx_labels_dict[task_name] = [example2label(ex) for ex in examples]
                str_labels_dict[task_name] = [ex.label for ex in examples]
            else:
                ValueError("labels required for eval mode")
    else:
        ValueError("Mode must be either inference or eval")

    if examples[0].text_b is None:
        sentences = [example.text_a for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]

    return idx_labels_dict, str_labels_dict, sentences
