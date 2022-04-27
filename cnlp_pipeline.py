import os

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import TaggingPipeline, get_sentences_and_labels

from .cnlp_processors import cnlp_processors, cnlp_output_modes, cnlp_compute_metrics, tagging, relex, classification

from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser

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

    pipeline_dict = {}
    
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
             
            pipeline_dict[task_name] = TaggingPipeline(
                model=model,
                tokenizer=tokenizer,
                task_processor=task_processor
            )

def get_sentences_and_labels(task_processor, in_file : str, mode : str):
    label_list = task_processor.get_labels()
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
    else:
        ValueError("Mode must be either inference or eval")
        
    label_map = {label : i for i, label in enumerate(label_list)}
    def example2label(example):
        return [label_map[label] for label in example.label]
    labels = [example2label(example) for example in examples]
    
    if examples[0].text_b is None:
        sentences = [example.text_a.split(' ') for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]
        
    return labels, sentences



            
def evaluation(pipeline_args : Tuple[DataClass, ...]):
    # holding off on this for now
    pass

if __name__ == "__main__":
    main()
