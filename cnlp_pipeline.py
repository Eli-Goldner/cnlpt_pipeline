from dataclasses import dataclass, field

from .cnlp_pipeline_utils import TaggingPipeline, get_sentences_and_labels

from transformers import HfArgumentParser

modes=["inf", "eval"]

@dataclass
class PipelineArguments:
    """
    Arguments pertaining to the models, mode, and data used by the pipeline.
    """
    model_dir: str = field(
        metadata={
            "help" : (
                "Path where each entity model is stored in a folder named after its tag suffix,"
                "e.g. 'med' for 'B-med I-med', and the relation extraction model stored in a folder named 'relex'"
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
    
def main():
    parser = HfArgumentParser((PipelineArguments))
