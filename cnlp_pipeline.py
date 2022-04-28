import os
import re

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import TaggingPipeline, get_sentences_and_labels

from .cnlp_processors import cnlp_processors, cnlp_output_modes, cnlp_compute_metrics, tagging, relex, classification

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
    parser = HfArgumentParser((PipelineArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        pipeline_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        pipeline_args = parser.parse_args_into_dataclasses()

        
    if pipeline_args.mode == "inf":
        inference()
    elif pipeline_args.mode == "eval":
        evaluation()
    else:
        ValueError("Invalid pipe mode!")

def inference(pipeline_args : Tuple[DataClass, ...]):
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    tokenizer = AutoTokenizer.from_pretrained(
        pipeline_args.tokenizer,
        # cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        additional_special_tokens=['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
    )

    taggers_dict = {}
    out_model_dict = {}
    
    for file in os.listdir(pipeline_args.models_dir):
        model_dir = os.path.join(pipeline_args.models_dir, file)
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
                pipeline_dict[task_name] = TaggingPipeline(
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
    _, sentences = get_sentences_and_labels(
        in_file = pipeline_args.in_file,
        mode="inf",
    )
    
    annotated_sents = assemble(
        sentences,
        pipeline_dict,
        pipeline_args.axis_task,
    )
    
    out_batch_encoding = tokenizer(
        annotated_sents,
        max_length=128,   # this and below two options need to be generalized   
        padding = "max_length",
        truncation=True,
        is_split_into_words=True,
    )
    
    for model in out_model_dict.values():
        logits = model(**out_batch_encoding)
        # more to be written here

            
def evaluation(pipeline_args : Tuple[DataClass, ...]):
    # holding off on this for now

    def partial_match(s1, s2):
        part = min(len(s1), len(s2))
        return s1[:part] == s2[:part]

    
    re.sub(r"</?a[1-2]>", "", new_sent)

    pass

        

def assemble(sentences, pipeline_dict, axis_task):
    return list(
        chain(
            *[_assemble(sent, pipeline_dict, axis_task) for sent
              in sentences]
        )
    )

def _assemble(sentence, pipeline_dict, axis_task):
    axis_ann = pipeline_dict[axis_task](sentence)
    sig_ann_ls = []
    for task, pipeline in pipeline_dict.items():
        if task != axis_task:
            sig_ann_ls += pipeline(sentence)
    return merge_annotations(axis_ann, sig_ann_ls, sentence)

def merge_annotations(axis_ann, sig_ann_ls, sentence):
    merged_annotations = []
    for sig_ann in sig_ann_ls:
        raw_partitions = get_partitions(axis_ann, sig_ann)
        anafora_tagged = get_anafora_tags(raw_partitions, sentences)
        merged_annotations += anafora_tagged
            
def get_partitions(axis_ann, sig_ann): 
    def tag2idx(t1, t2):
        if t1 != 'O' and t2 != 'O':
            ValueError("Overlapping tags!")
        elif t1 != 'O':
            return 1
        elif t2 != 'O':
            return 2
        else:
            ValueError(f"Something really strange happened: {t1}, {t2}")
    return map(tag2idx, zip(axis_ann, sig_ann))


def get_sentences_and_labels(task_processor=None, in_file : str, mode : str): 
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
        sentences = [example.text_a.split(' ') for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]
        
    return labels, sentences


if __name__ == "__main__":
    main()
