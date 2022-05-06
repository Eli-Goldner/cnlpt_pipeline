# cnlpt_pipeline

Designed to work with [CNLP transformers](https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers/). To use, copy the contents of this repository into the `src/cnlpt/` folder of a clone of the latest main version of `cnlp_transformers`.

## Instructions

Call via `python -m cnlpt.cnlp_pipeline`:
```
usage: cnlp_pipeline.py [-h] --models_dir MODELS_DIR --in_file IN_FILE
                        [--mode MODE] [--axis_task AXIS_TASK]

optional arguments:
  -h, --help            show this help message and exit
  --models_dir MODELS_DIR
                        Path where each entity model is stored in a folder
                        named after its corresponding cnlp_processor, models
                        with a 'tagging' output mode will be run first followed
                        by models with a 'classification' ouput mode over the
                        assembled data (default: None)
  --in_file IN_FILE     Path to file, with one raw sentenceper line in the
                        case of inference, and one <label> <annotated
                        sentence> per line in the case of evaluation (default:
                        None)
  --mode MODE           Use mode for full pipeline, inference, which outputs
                        annotated sentencesand their relation, or eval, which
                        outputs metrics for a provided set of samples
                        (requires labels) (default: inf)
  --axis_task AXIS_TASK
                        key of the task in cnlp_processors which generates the
                        tag that will map to <a1> <mention> </a1> in pairwise
                        annotations (default: dphe_med)

```


## Structure