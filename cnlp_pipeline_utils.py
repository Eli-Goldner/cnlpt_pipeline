import os

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

from itertools import chain, groupby, tee

SPECIAL_TOKENS = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']


# Get the model (by task_name) pairs for each instance e.g.
# (task : dphe_rel) label: med-dosage -> model_pair : (dphe_med, dphe_dosage)
def get_model_pairs(str_labels_dict, taggers_dict):
    model_pairs = {}
    model_names = [key for key, value in taggers_dict.items()]
    # Used to get the relevant models from the label name,
    # e.g. med-dosage -> med, dosage -> dphe-med, dphe-dosage,
    # -> taggers_dict[dphe-med], taggers_dict[dphe-dosage]

    def partial_match(s1, s2):
        part = min(len(s1), len(s2))
        return s1[:part] == s2[:part]
    for task_name, labels in str_labels_dict.items():
        model_pairs[task_name] = []
        for label in labels:
            axis_tag, sig_tag = label.split('-')
            # Get the first (only) model/task name
            # the end of which matches the axis tag
            axis_model = list(
                filter(
                    lambda x: partial_match(x.split('_')[-1], axis_tag),
                    model_names
                )
            )[0]
            # Get the first (only) model/task name
            # the end of which matches the axis tag
            sig_model = list(
                filter(
                    lambda x: partial_match(x.split('_')[-1], sig_tag),
                    model_names
                )
            )[0]
            model_pairs[task_name].append((axis_model, sig_model))
        assert len(model_pairs[task_name]) == len(str_labels_dict[task_name]), (
            "Wrong lengths"
            f"task model pairs : {model_pairs[task_name]}"
            f"task labels : {str_labels_dict[task_name]}"
        )
    return model_pairs


# Get dictionary of entity tagging models/pipelines
# and relation extraction models/pipelines
# both indexed by task names
def model_dicts(models_dir):
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
                out_model_dict[task_name] = ClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor
                )
            # Tasks other than tagging and sentence/relation classification
            # not supported for now since I wasn't sure how to fit them in
            else:
                ValueError(
                    (
                        "output mode "
                        f"{cnlp_output_modes[task_name]} "
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
    return list(
        chain(
            *[_assemble(sent, taggers_dict, axis_task, mode='inf') for sent
              in sentences]
        )
    )


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
        if any(filter(lambda x: x.startswith('B'), axis_ann)):
            sig_ann_ls = []
            for task, pipeline in taggers_dict.items():
                # Don't rerun the axial pipeline
                if task != axis_task:
                    sig_out = pipeline(sentence)
                    sig_ann_ls.append(sig_out)
            ann_sents = merge_annotations(axis_ann, sig_ann_ls, sentence)
    elif mode == 'eval':
        # For eval, we need to run both taggers regardless of
        # whether they miss
        ann_sents = []
        sig_ann_ls = []
        for task, pipeline in taggers_dict.items():
            if task != axis_task:
                sig_out = pipeline(sentence)
                sig_ann_ls.append(sig_out)
        ann_sents = merge_annotations(axis_ann, sig_ann_ls, sentence)
    else:
        ValueError("Invalid processing mode: {model}")
    return ann_sents


# Given a raw sentence, an axial entity tagging,
# and a non-axial entity tagging,
# return the sentence with XML tags
# around the entities
# - this is another wrapper function where
# we handle looping and some optimizations
def merge_annotations(axis_ann, sig_ann_ls, sentence):
    merged_annotations = []
    for sig_ann in sig_ann_ls:
        raw_partitions = get_partitions(axis_ann, sig_ann)
        anafora_tagged = get_anafora_tags(raw_partitions, sentence)
        # Filter out non-axial tags that didn't find any annotations
        # as well
        if anafora_tagged is not None:
            merged_annotations.append(" ".join(anafora_tagged))
    return merged_annotations


# Shamelessly grabbed from https://stackoverflow.com/a/57293089
def get_intersect(ls1, ls2):
     m1, m2 = tee(merge(ls1, ls2, key=lambda k: k[0]))
     next(m2, None)
     out = []
     for v, g in groupby(zip(m1, m2), lambda k: k[0][1] < k[1][0]):
             if not v:
                     l = [*g][0]
                     inf = max(i[0] for i in l)
                     sup = min(i[1] for i in l)
                     if inf != sup:
                         out.append((inf, sup))
     return out

# Turn two annotations into a list of integers,
# used for fast grouping of indices
# in get_anafora_tags
def get_partitions(axis_ann, sig_ann):
    assert len(axis_ann) == len(sig_ann), (
        "Taggnings are not aligned: \n"
        f"Axis : {axis_ann}"
        f"Signature : {sig_ann}"
    )

    def tag2idx(tag_pair):
        t1, t2 = tag_pair
        # Ensure no overlapping annotations
        if t1 != 'O' and t2 != 'O':
            ValueError("Overlapping tags!")
        # 1 identifies the first tag
        elif t1 != 'O':
            return 1
        # 2 identifies the second
        elif t2 != 'O':
            return 2
        # 0 identifies no tag
        elif t1 == 'O' and t2 == 'O':
            return 0
    return map(tag2idx, zip(axis_ann, sig_ann))


# Given a list of integers over {0,1,2}
# and an unannotated sentence,
# return the sentence with <a1> X </a1>
# where X is the span of the split sentence
# corresponding to the span of 1's in the list of integers
# similarly for 2 and <a2>
def get_anafora_tags(raw_partitions, sentence):
    span_begin = 0
    annotated_list = []
    split_sent = ctakes_tok(sentence)
    sig_seen = False
    for tag_idx, span_iter in groupby(raw_partitions):
        span_end = len(list(span_iter)) + span_begin
        # Ran into this issue when given a sentences
        # with nothing but mentions
        if span_end > 0:
            span_end = span_end - 1
        # Get the span of the split sentence
        # which is aligned with the current
        # span of the same integer
        span = split_sent[span_begin:span_end]
        ann_span = span
        # Tags are done by type, not order of appearence
        if tag_idx == 1:
            ann_span = ['<a1>'] + span + ['</a1>']
        elif tag_idx == 2:
            ann_span = ['<a2>'] + span + ['</a2>']
            # We know a priori from earlier filtering
            # <a1> is taken care of, here we make
            # sure <a2> is hit
            sig_seen = True
        annotated_list.extend(ann_span)
        span_begin = span_end
    return annotated_list if sig_seen else None


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

        pipe_output = out_pipe(
            reannotated_sents[out_task],
            # Again, classification pipelines
            # take tokenizer args during __call__
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        out_labels = [out_label_map[record['label']] for record in pipe_output]
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
