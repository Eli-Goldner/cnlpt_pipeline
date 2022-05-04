import types
from typing import List, Optional, Union
from itertools import groupby

import numpy as np

from transformers.utils import add_end_docstrings
from transformers.pipelines.base import (
    PIPELINE_INIT_ARGS,
    ArgumentHandler,
    Dataset,
    Pipeline,
)

from transformers.data.processors.utils import DataProcessor

from .__init__ import ctakes_tok

class TaggingArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if (
                inputs is not None and isinstance(inputs, (list, tuple))
                and len(inputs) > 0
        ):
            inputs = list(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
        elif (
                Dataset is not None and isinstance(inputs, Dataset)
                or isinstance(inputs, types.GeneratorType)
        ):
            return inputs, None
        else:
            raise ValueError("At least one input is required.")

        return inputs


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        ignore_labels (`List[str]`, defaults to `["O"]`):
            A list of labels to ignore.
        task_processpr (`DataProcessor`, defaults to `None`):
            cnlp_processor used to pull the task labels.
    """,
)
class TaggingPipeline(Pipeline):
    """
    NER/Token Classification/Entity Tagging pipeline adapted for cnlpt models
    """

    default_input_names = "sequences"

    def __init__(
            self,
            args_parser=TaggingArgumentHandler(),
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._args_parser = args_parser

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        task_processor: Optional[DataProcessor] = None,
    ):
        preprocess_params = {}

        postprocess_params = {}

        if task_processor is not None:
            postprocess_params["task_processor"] = task_processor
        elif "task_processor" not in self._postprocess_params:
            raise ValueError("Task_processor was never initialized")

        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Flesh out - see HF code for format
        """
        # If an arg parsers results are never called
        # in the forest but someone comments it
        # does it still have side effects?
        # _inputs = self._args_parser(inputs, **kwargs)
        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence):
        model_inputs = self.tokenizer(
            ctakes_tok(sentence),
            max_length=128,  # UN-HARDCODE
            return_tensors=self.framework,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        # This is what we use for cnlpt/cTAKES based tagging
        # rather than an aggregation strategy
        model_inputs["word_ids"] = model_inputs.word_ids()

        return model_inputs

    def _forward(self, model_inputs):
        word_ids = model_inputs.pop("word_ids")
        if self.framework == "tf":
            logits = self.model(model_inputs.data)[0]
        else:
            logits = self.model(**model_inputs)[0]

        return {
            "logits": logits,
            "word_ids": word_ids,
            **model_inputs,
        }

    def postprocess(
            self,
            model_outputs,
            task_processor: DataProcessor,
            ignore_labels=None):
        logits = model_outputs["logits"][0].numpy()
        input_ids = model_outputs["input_ids"][0]
        word_ids = model_outputs["word_ids"]

        if task_processor is None:
            raise ValueError("You guessed it!")

        label_list = task_processor.get_labels()

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        return self.get_annotation(input_ids, word_ids, scores[0], label_list)

    def get_annotation(
            self,
            input_ids: np.ndarray,
            word_ids,
            scores: np.ndarray,
            label_list,
    ) -> List[dict]:
        final_tags = []
        assert len(word_ids) == len(scores), (
            "Eq problem 1 \n"
            f"word_ids : {word_ids}"
            f"scores : {scores}"
        )
        assert len(word_ids) == len(input_ids), (
            "Eq problem 2"
            f"word_ids : {word_ids}"
            f"input_ids : {input_ids}"
        )

        prev_word_id = None

        for idx, token_scores in enumerate(scores):
            word_id = word_ids[idx]
            # reason for three conditions allows for the last tag to get seen
            if word_id is None:
                prev_word_id = word_id
            elif word_id != prev_word_id:
                label_idx = token_scores.argmax()
                label = label_list[label_idx]
                final_tags.append(label)
                prev_word_id = word_id
            else:
                prev_word_id = word_id
        return final_tags
