import dataclasses
import sys
import os
import torch
import numpy as np


from tqdm.auto import tqdm

from typing import Callable, Dict, Optional, List, Union, Tuple

from torch.utils.data import DataLoader

from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer, AutoModel, EvalPrediction, default_data_collator
from .cnlp_processors import cnlp_processors, cnlp_output_modes, cnlp_compute_metrics, tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset, DataTrainingArguments

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from .clinical_pipelines import TaggingPipeline 

from transformers.utils import ExplicitEnum

class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


class CnlpPipeline:
    def __init__(
            self,
            model,
            tokenizer,
            task_name
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = cnlp_processors[task_name]()
        self.label_list = self.processor.get_labels()
        

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.
        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.
        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
            )
        elif isinstance(inputs, dict):
            return {name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
        elif isinstance(inputs, UserDict):
            return UserDict({name: self._ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple([self._ensure_tensor_on_device(item, device) for item in inputs])
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        else:
            return inputs

        
    def __call__(self, data_file, mode, batch_size=64):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        self.model.eval()

        labels, sentences = self.get_sentences_and_labels(data_file, mode)
        if mode == 'eval':
            assert labels is not None, "Labels required for evaluation"
            assert len(labels) == len(sentences), "Each sentence needs a label for evaluation"

            
    def preprocess(self, sentences):
            model_inputs = self.tokenizer(
                sentences,
                max_length = 128, # hardcoding cnlp_data DataTrainingArguments default max_seq_length for now
                padding = "max_length",
                truncation=True,
                is_split_into_words=True,
            )
                
    def get_sentences_and_labels(self, in_file : str, mode : str):
        if mode == 'inference':
            # 'test' let's us forget labels
            examples =  self.processor._create_examples(
                self.processor._read_tsv(in_file),
                "test"
            )
        elif mode == 'eval':
            # 'dev' lets us get labels without running into issues of downsampling
            examples = self.processor._create_examples(
                self.processor._read_tsv(in_file),
                "dev"
            )
        else:
            ValueError("Mode must be either inference or eval")

        label_map = {label : i for i, label in enumerate(self.label_list)}
        def example2label(example):
            return [label_map[label] for label in example.label]
        labels = [example2label(example) for example in examples]

        if examples[0].text_b is None:
            sentences = [example.text_a.split(' ') for example in examples]
        else:
            sentences = [(example.text_a, example.text_b) for example in examples]
        
        return labels, sentences

    def get_inference_context(self):
        inference_context = (
            torch.inference_mode if version.parse(torch.__version__) >= version.parse("1.9.0") else torch.no_grad
        )
        return inference_context

    def forward(self, model_inputs, **forward_params):
        with self.device_placement():
            if self.framework == "tf":
                model_inputs["training"] = False
                model_outputs = self._forward(model_inputs, **forward_params)
            elif self.framework == "pt":
                inference_context = self.get_inference_context()
                with inference_context():
                    model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
                    model_outputs = self._forward(model_inputs, **forward_params)
                    model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
            else:
                raise ValueError(f"Framework {self.framework} is not supported")
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, self.feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator
    
class CnlpTaggingPipeline(CnlpPipeline):
    def __init__(
            self,
            model,
            tokenizer,
            task_name
    ):
        super().__init__(model, tokenizer, task_name)
    
    def postprocess(self, model_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        logits = model_outputs["logits"][0] #Due to being wrapped
        input_ids = model_outputs["input_ids"]
        
        maxes, _ = torch.max(logits, dim=-1, keepdim=True)
        shifted_exp = torch.exp(logits - maxes)
        scores = shifted_exp / torch.sum(shifted_exp, dim=-1, keepdim=True)
        
        entities = []
        
        for sent_ids, sent_scores in zip(input_ids, scores):
            #pre_entities = self.gather_pre_entities(
            #    sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
            #)
            pre_entities = self.gather_pre_entities(
                sent_ids, sent_scores, aggregation_strategy
            )
        
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            sent_entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
            entities += sent_entities
        return entities

    def gather_pre_entities(
        #tokenizer,
        self,
        #sentence: str,
        input_ids, #: np.ndarray,
        scores, #: np.ndarray,
        #offset_mapping: Optional[List[Tuple[int, int]]],
        #special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        decoded_output = self.tokenizer.decode(
            [int(idx) for idx in input_ids],
            clean_up_tokenization_spaces=False
        )
        sentence = decoded_output.split("</s>")[0][4:] 
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens, they should only occur
            # at the sentence boundaries since we're not encoding pairs of
            # sentences so we don't have to keep track of those.
            #if special_tokens_mask[idx]:
            #    continue
            
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            
            pre_entity = {
                "word": word,
                "scores": token_scores,
                #"start": start_ind,
                #"end": end_ind,
                "index": idx,
                #"is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate(
            self,
            pre_entities: List[dict],
            aggregation_strategy: AggregationStrategy
    ) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                #entity_idx = pre_entity["scores"].argmax()
                entity_idx = int(pre_entity["scores"].argmax())
                #pre_entity["scores"][entity_idx]
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.labels[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    #"start": pre_entity["start"],
                    #"end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)
            
        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities(entities)

    def aggregate_word(
            self,
            entities: List[dict],
            aggregation_strategy: AggregationStrategy
    ) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def aggregate_words(
            self,
            entities: List[dict],
            aggregation_strategy: AggregationStrategy
    ) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.
        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        if aggregation_strategy in {
                AggregationStrategy.NONE,
                AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # Last item
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.
        Args:
        entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]
    
        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # It's not in B-, I- format
            # Default to I- for continuation.
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.
        Args:
        entities (`dict`): The entities predicted by the pipeline.
        """
        
        entity_groups = []
        entity_group_disagg = []
        
        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # If the current entity is similar and adjacent to the previous entity,
            # append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" prefixes
            # Shouldn't merge if both entities are B-type
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            if tag == last_tag and bi != "B":
                # Modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # If the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        return entity_groups
 
