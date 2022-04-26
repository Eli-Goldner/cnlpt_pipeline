import types
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.utils import ExplicitEnum, add_end_docstrings, is_tf_available, is_torch_available
from transformers.pipeline.base import PIPELINE_INIT_ARGS, ArgumentHandler, Dataset, Pipeline


if is_tf_available():
    from transformers.models.auto.modeling_tf_auto import TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

if is_torch_available():
    from transformers.models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

class TaggingArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):

        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        elif Dataset is not None and isinstance(inputs, Dataset) or isinstance(inputs, types.GeneratorType):
            return inputs, None
        else:
            raise ValueError("At least one input is required.")
        
        task_processor = kwargs.get("task_processor")
        if task_processor is None:
            raise ValueError("Pipelining with Cnlpt requires the task's cnlp_processor")
        return inputs, task_processor


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        CHANGE TO FIT WHAT WE NEED
        ignore_labels (`List[str]`, defaults to `["O"]`):
            A list of labels to ignore.
        grouped_entities (`bool`, *optional*, defaults to `False`):
            DEPRECATED, use `aggregation_strategy` instead. Whether or not to group the tokens corresponding to the
            same entity together in the predictions or not.
        aggregation_strategy (`str`, *optional*, defaults to `"none"`):
            The strategy to fuse (or not) tokens based on the model prediction.
                - "none" : Will simply not do any aggregation and simply return raw results from the model
                - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C,
                  I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D",
                  "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as
                  different entities. On word based languages, we might end up splitting words undesirably : Imagine
                  Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity":
                  "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages
                  that support that meaning, which is basically tokens separated by a space). These mitigations will
                  only work on real words, "New york" might still be tagged with two different entities.
                - "first" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Words will simply use the tag of the first token of the word when there
                  is ambiguity.
                - "average" : (works only on word based models) Will use the `SIMPLE` strategy except that words,
                  cannot end up with different tags. scores will be averaged first across tokens, and then the maximum
                  label is applied.
                - "max" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Word entity will simply be the token with the maximum score.
    """,
)


class TaggingPipeline(Pipeline):
    """
    Ripped off from the eponymous HF code with some (in hindsight...) trivial tweaks for cnlpt

    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.
    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).
    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    """

    default_input_names = "sequences"

    def __init__(self, args_parser=TaggingArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        #self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        task_processor = None, # try to write the type hint at some point
    ):
        preprocess_params = {}

        postprocess_params = {} 

        if task_processor is not None:
            postprocess_params["task_processor"] = task_processor
        else:
            raise ValueError("Task_processor required!")

        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        ADJUST LATER
        Classify each token of the text(s) given as inputs.
        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of texts) for token classification.
        Return:
            A list or a list of list of `dict`: Each result comes as a list of dictionaries (one for each token in the
            corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
            the following keys:
            - **word** (`str`) -- The token/word classified.
            - **score** (`float`) -- The corresponding probability for `entity`.
            - **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
              *aggregation_strategy* is not `"none"`.
            - **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
              token in the sentence.
            - **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
            - **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
        """

        _inputs, task_processor = self._args_parser(inputs, **kwargs)
        if offset_mapping:
            kwargs["task_processor"] = task_processor

        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence):
        model_inputs = self.tokenizer(
            sentences,
            max_length = 128, # hardcoding cnlp_data DataTrainingArguments default max_seq_length for now
            # maybe this means that other args would be preprocess params?
            padding = "max_length",
            truncation=True,
            is_split_into_words=True,
        )
        
        model_inputs["sentence"] = sentence
        # This is what we use for cnlpt/cTAKES based tagging
        # rather than an aggregation strategy
        model_inputs["word_ids"] = model_inputs.word_ids()
        
        return model_inputs

    def _forward(self, model_inputs):
        # Forward
        sentence = model_inputs.pop("sentence")
        word_ids = model_inputs.pop("word_ids")
        if self.framework == "tf":
            logits = self.model(model_inputs.data)[0]
        else:
            logits = self.model(**model_inputs)[0]

        return {
            "logits": logits,
            "sentence": sentence,
            "word_ids": word_ids, 
            **model_inputs,
        }

    def postprocess(self, model_outputs, task_processor=None, ignore_labels=None):
        # Makes sense why they have it but we need to keep O's
        # nevertheless, holding on to the parameter in case posterity
        # finds a use for it
        # if ignore_labels is None:
        #    ignore_labels = ["O"]
        logits = model_outputs["logits"][0].numpy()
        sentence = model_outputs["sentence"]
        input_ids = model_outputs["input_ids"][0]
        word_ids = model_outputs["word_ids"][0]
        

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        # Trying it their way for ease of reading
        # pre-entities roughly are tokens
        pre_entities = self.gather_pre_entities(
            input_ids, word_ids, scores #, offset_mapping, special_tokens_mask, aggregation_strategy - don't use/need these
        )

        # For aggregation we'll use Dongfang's logic from cnlp_data
        grouped_entities = self.aggregate(pre_entities) #, aggregation_strategy)
        # Filter anything that is in self.ignore_labels
        entities = [
            entity
            for entity in grouped_entities
            if entity.get("entity", None) not in ignore_labels
            and entity.get("entity_group", None) not in ignore_labels
        ]
        return entities

    def gather_pre_entities(
            self,
            input_ids: np.ndarray,
            word_ids,
            scores: np.ndarray,
    ) -> List[dict]:
        pre_entities = []
        for idx, token_scores in enumerate(scores):

            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            pre_entity = {
                "word": word,
                "index": idx,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(self, pre_entities: List[dict]) -> List[dict]:
        entities = []
        for pre_entity in pre_entities:
            entity_idx = pre_entity["scores"].argmax()
            score = pre_entity["scores"][entity_idx]
            entity = {
                "entity": self.model.config.id2label[entity_idx],
                "score": score,
                "index": pre_entity["index"],
                "word": pre_entity["word"],
                "start": pre_entity["start"],
                "end": pre_entity["end"],
            }
            entities.append(entity)
        return entities
        
