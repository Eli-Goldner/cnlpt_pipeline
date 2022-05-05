import os
import re
import torch
import sys
import types
from typing import List, Optional, Union

from dataclasses import dataclass, field

from .cnlp_processors import (
    cnlp_processors,
    cnlp_output_modes,
    tagging,
    classification
)

from .pipelines.tagging import TaggingPipeline
from .pipelines.classification import ClassificationPipeline
from .pipelines import ctakes_tok

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser

from itertools import chain, groupby

SPECIAL_TOKENS = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']

def get_model_pairs(labels, taggers_dict):
    model_pairs = []
    model_names = [key for key, value in taggers_dict.items()]
    model_suffixes = [name.split('_')[-1] for name in model_names]
    def partial_match(s1, s2):
        part = min(len(s1), len(s2))
        return s1[:part] == s2[:part]
    for label in labels:
        axis_tag, sig_tag = label.split('-')
        axis_model = list(
            filter(
                lambda x : partial_match(x.split('_')[-1], axis_tag),
                model_names
            )
        )[0]
        sig_model = list(
            filter(
                lambda x : partial_match(x.split('_')[-1], sig_tag),
                model_names
            )
        )[0]
        model_pairs.append((axis_model, sig_model))
    assert len(model_pairs) == len(labels), "Wrong lengths"
    return model_pairs


def model_dicts(models_dir):    
    taggers_dict = {}
    out_model_dict = {}
    
    for file in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, file)
        task_name = str(file)
        if os.path.isdir(model_dir) and task_name in cnlp_processors.keys():

            config = AutoConfig.from_pretrained(
                model_dir,
            )

            model = CnlpModelForClassification.from_pretrained(
                model_dir,
                config=config,
            )

            # right now assume roberta
            tokenizer = AutoTokenizer.from_pretrained(
                #pipeline_args.tokenizer,
                model_dir,
                # cache_dir=model_args.cache_dir,
                add_prefix_space=True,
                additional_special_tokens=SPECIAL_TOKENS,
            )

            
            task_processor = cnlp_processors[task_name]()
            
            if cnlp_output_modes[task_name] == tagging:
                taggers_dict[task_name] = TaggingPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor
                )
            elif cnlp_output_modes[task_name] == classification:
                out_model_dict[task_name] = ClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor
                )
            else:
                ValueError(
                    f"output mode {cnlp_output_modes[task_name]} not currently supported"
                )
    return taggers_dict, out_model_dict 

def assemble(sentences, taggers_dict, axis_task):
    return list(
        chain(
            *[_assemble(sent, taggers_dict, axis_task) for sent in sentences]
        )
    )

def _assemble(sentence, taggers_dict, axis_task):
    axis_pipe = taggers_dict[axis_task]
    axis_ann = axis_pipe(sentence)
    sig_ann_ls = []
    for task, pipeline in taggers_dict.items():
        if task != axis_task:
            sig_out= pipeline(sentence)
            sig_ann_ls.append(sig_out)
    return merge_annotations(axis_ann, sig_ann_ls, sentence)

def merge_annotations(axis_ann, sig_ann_ls, sentence):
    merged_annotations = []
    for sig_ann in sig_ann_ls:
        raw_partitions = get_partitions(axis_ann, sig_ann)
        anafora_tagged = get_anafora_tags(raw_partitions, sentence)
        if anafora_tagged is not None:
            merged_annotations.append(" ".join(anafora_tagged))
    return merged_annotations
            
def get_partitions(axis_ann, sig_ann):
    assert len(axis_ann) == len(sig_ann), "make sure"
    def tag2idx(tag_pair):
        t1, t2 = tag_pair
        if t1 != 'O' and t2 != 'O':
            ValueError("Overlapping tags!")
        elif t1 != 'O':
            return 1
        elif t2 != 'O':
            return 2
        elif t1 == 'O' and t2 == 'O':
            return 0
    return map(tag2idx, zip(axis_ann, sig_ann))

def get_anafora_tags(raw_partitions, sentence):
    span_begin = 0
    annotated_list = []
    split_sent = ctakes_tok(sentence)
    axis_seen, sig_seen = False, False 
    for tag_idx, span_iter in groupby(raw_partitions):
        span_end = len(list(span_iter)) + span_begin
        span = split_sent[span_begin:span_end]
        ann_span = span
        if tag_idx == 1:
            ann_span = ['<a1>'] + span + ['</a1>']
            axis_seen = True
        elif tag_idx == 2:
            ann_span = ['<a2>'] + span + ['</a2>']
            sig_seen = True
        annotated_list.extend(ann_span)
        span_begin = span_end
    return annotated_list if axis_seen and sig_seen else None


def get_eval_predictions(
        model_pairs,
        deannotated_sents,
        taggers_dict,
        out_model_dict,
        axis_task,
):
    reannotated_sents = []
    for sent, pair in zip(deannotated_sents, model_pairs):
        tagger_pair_dict = {key: taggers_dict[key] for key in pair}
        reann_sent_ls = _assemble(sent, tagger_pair_dict, axis_task)
        assert len(reann_sent_ls) == 1, "ASDASDDASDASDASD"
        reannotated_sents.append(reann_sent_ls[0])

    """
    softmax = torch.nn.Softmax(dim=1)
    ann_encoding = tokenizer(
            reannotated_sents,
            max_length=128,   # UN-HARDCODE
            return_tensors='pt',
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )
    """
    
    out_labels = None
    
    for out_task, out_pipe in out_model_dict.items():

        out_task_processor = cnlp_processors[out_task]()
        out_task_labels = out_task_processor.get_labels()

        out_label_map = {label : i for i, label in enumerate(out_task_labels)}
        
        pipe_output = out_pipe(
            reannotated_sents,
            max_length=128,   # UN-HARDCODE
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        ) 
        # labels = [out_task_labels[idx.item()] for idx in out_label_idxs]
        
        # for label, ann_sent in zip(labels, reannotated_sents):
        #    print(f"{label} : {ann_sent}")
        # out_labels = [idx.item() for idx in out_label_idxs]
        print(pipe_output)
        out_labels = [out_label_map[record['label']] for record in pipe_output]
    return out_labels
    
            
def get_sentences_and_labels(in_file : str, mode : str, task_processor):
    idx_labels, str_labels = None, None
    if mode == "inf":
        # 'test' let's us forget labels
        examples =  task_processor._create_examples(
            task_processor._read_tsv(in_file),
            "test"
        )
    elif mode == "eval":
        # 'dev' lets us get labels without running into issues of downsampling
        examples = task_processor._create_examples(
            task_processor._read_tsv(in_file),
            "dev"
        )
        
        label_list = task_processor.get_labels()
        label_map = {label : i for i, label in enumerate(label_list)}
        def example2label(example):
            if isinstance(example.label, list):
                return [label_map[label] for label in example.label]
            else:
                return label_map[example.label]

        if examples[0].label:
            idx_labels = [example2label(example) for example in examples]
            str_labels = [example.label for example in examples]
        else:
            ValueError("labels required for eval mode")
    else:
        ValueError("Mode must be either inference or eval")
        
      
    if examples[0].text_b is None:
        # sentences = [example.text_a.split(' ') for example in examples]
        # pipeline freaks out if it's already split
        sentences = [example.text_a for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]
        
    return idx_labels, str_labels, sentences
