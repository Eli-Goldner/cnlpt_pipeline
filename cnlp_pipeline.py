import os
import re
import torch
import sys

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import TaggingPipeline

from .cnlp_processors import cnlp_processors, cnlp_output_modes, cnlp_compute_metrics, tagging, relex, classification

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser

from itertools import chain, groupby

modes=["inf", "eval"]

@dataclass
class PipelineArguments:
    """
    Arguments pertaining to the models, mode, and data used by the pipeline.
    """
    models_dir: str = field(
        metadata={
            "help" : (
                "Path where each entity model is stored in a folder named after its cnlp_processor,"
                "models with a 'tagging' output mode will be run first"
                "followed by models with a 'classification' ouput mode over the assembled data"
            )
        }
    )
    in_file: str = field(
        metadata = {
            "help" : (
                "Path to file with one raw sentence per line in the case of inference,"
                " and one <label>\t<annotated sentence> per line in the case of evaluation"
            )
        }
    )
    mode: str = field(
        default = "inf",
        metadata = {
            "help" : (
                "Use mode for full pipeline, inference which outputs annotated sentences and their relation,"
                "or eval which outputs metrics for a provided set of samples (requires labels)"
            )
        }
    )
    tokenizer: str = field(
        default="roberta-base",
        metadata = {
            "help" : (
                "At the moment, assuming everything has been trained using the same tokenizer."
                "Figuring out ways around this later might be annoying, including saving each tokenizer"
                "in each model dir"
            )
        }
    )
    axis_task: str = field(
        default = "dphe_med",
        metadata = {
            "help" : (
                "key of the task in cnlp_processors which generates the tag that will map to <a1> <mention> </a1> in pairwise annotations"
            )
        }
    )
    
def main():
    parser = HfArgumentParser(PipelineArguments)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.

        #the ',' is to deal with unary tuple weirdness
        pipeline_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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
        additional_special_tokens=['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
    )


    taggers_dict, out_model_dict = model_dicts(pipeline_args.models_dir, tokenizer)

    sent_processor = None

    for key in out_model_dict.keys():
        sent_processor = key
    
    _, sentences = get_sentences_and_labels(
        in_file = pipeline_args.in_file,
        mode="inf",
        task_processor = cnlp_processors[sent_processor](),
    )
    
    annotated_sents = assemble(
        sentences,
        taggers_dict,
        pipeline_args.axis_task,
    )

    softmax = torch.nn.Softmax(dim = 1)
    
    for ann_sent in annotated_sents:
        ann_encoding = tokenizer(
            ann_sent.split(' '),
            max_length=128,   # this and below two options need to be generalized
            return_tensors='pt',
            padding = "max_length",
            truncation=True,
            is_split_into_words=True,
        )
        
        for out_task, model in out_model_dict.items():
            out_task_processor = cnlp_processors[out_task]()
            out_task_labels = out_task_processor.get_labels()
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

    labels, sentences = get_sentences_and_labels(
        in_file = pipeline_args.in_file,
        mode="inf",
    )

    # strip the sentence of tags
    re.sub(r"</?a[1-2]>", "", new_sent)

    pass


def classify(labels, sentences):
    def partial_match(s1, s2):
        part = min(len(s1), len(s2))
        return s1[:part] == s2[:part]
    

def model_dicts(models_dir, tokenizer):    
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

            task_processor = cnlp_processors[task_name]()
            
            if cnlp_output_modes[task_name] == tagging:
                taggers_dict[task_name] = TaggingPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    task_processor=task_processor
                )
            elif cnlp_output_modes[task_name] == classification:
                out_model_dict[task_name] = model
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
        else:
            ValueError(f"Something really strange happened: {t1}, {t2}")
    return map(tag2idx, zip(axis_ann, sig_ann))

def get_anafora_tags(raw_partitions, sentence):
    span_begin = 0
    annotated_list = []
    split_sent = sentence.split(' ') 
    for tag_idx, span_iter in groupby(raw_partitions):
        span_end = len(list(span_iter)) + span_begin
        span = split_sent[span_begin:span_end]
        ann_span = span
        if tag_idx == 1:
            ann_span = ['<a1>'] + span + ['</a1>'] 
        elif tag_idx == 2:
            ann_span = ['<a2>'] + span + ['</a2>']
        annotated_list.extend(ann_span)
        span_begin = span_end
    return annotated_list
        
def get_sentences_and_labels(in_file : str, mode : str, task_processor): 
    if mode == "inf":
        # 'test' let's us forget labels
        examples =  task_processor._create_examples(
            task_processor._read_tsv(in_file),
            "test"
        )
        labels = None
    elif mode == "eval":
        # 'dev' lets us get labels without running into issues of downsampling
        examples = task_processor._create_examples(
            task_processor._read_tsv(in_file),
            "dev"
        )
        label_list = task_processor.get_labels()
        label_map = {label : i for i, label in enumerate(label_list)}
        def example2label(example):
            return [label_map[label] for label in example.label]

        if examples[0].label:
            labels = [example2label(example) for example in examples]
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
        
    return labels, sentences


if __name__ == "__main__":
    main()
