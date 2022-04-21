from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class



import logging
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument()


model = CnlpModelForClassification.from_pretrained(
    model_args.encoder_name,
    config=config,
    class_weights=None if train_dataset is None else train_dataset.class_weights,
    final_task_weight=training_args.final_task_weight,
    freeze=training_args.freeze,
    bias_fit=training_args.bias_fit,
    argument_regularization=training_args.arg_reg)
