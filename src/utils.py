import dataclasses
import torch
from typing import List, Dict, TypedDict
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os

from src.prompts import (
    EXTRACT_ENTITY_TYPE_PROMPT,
    REFINEMENT_LABELS_PROMPT,
    CONFIDENCE_SCORE_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    CONFIDENCE_SCORE_PROMPT,
    REFINEMENT_LABELS_PROMPT_FEWREL,
    CONFIDENCE_SCORE_PROMPT_FEWREL_SEP,
    RELATION_EXTRACTION_PROMPT_FEWREL,
    EXTRACT_ENTITY_TYPE_PROMPT_WIKI,
    REFINEMENT_LABELS_PROMPT_WIKI,
    CONFIDENCE_SCORE_PROMPT_WIKI_SEP,
    CONFIDENCE_SCORE_PROMPT_WIKI_JOINT,
    RELATION_EXTRACTION_PROMPT_WIKI,
    EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_WIKI_PROMPT,
    EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_FEWREL_PROMPT,
    CONFIDENCE_SCORE_PROMPT_FEWREL_JOINT,
    
)

class ModelType(Enum):
    PROTECTED = "protected"
    OPEN = "open"
    GPT = "gpt"
    
@dataclasses.dataclass
class SepConfig:
    max_tokens_extract: int
    max_tokens_refine: int
    max_tokens_label_mapping: int
    max_tokens_confidence: int
    max_tokens_relation_extraction: int
    extract_prompt: str
    refine_prompt: str
    confidence_prompt: str
    relation_extraction_prompt: str
    extract_command: str = "Entity Types: "
    refine_command: str = "Refined Relation Labels: "
    confidence_command: str = "Relationship Confidence Scores Sentences: "
    relation_extraction_command: str = "the most appropriate relationship is: "

@dataclasses.dataclass
class JointConfig:
    max_tokens_extract_refine: int
    max_tokens_confidence: int
    max_tokens_relation_extraction: int
    extract_refine_prompt: str
    confidence_prompt: str
    relation_extraction_prompt: str
    extract_refine_command: str = 'Refined relation labels: '
    confidence_command: str = 'Relationship Confidence Scores Sentences: '
    relation_extraction_command: str = 'the most appropriate relationship is: '
    
@dataclasses.dataclass
class TaskConfig:
    id: str
    max_tokens: int
    sep: SepConfig
    joint: JointConfig

class ModelConfigDict(TypedDict):
    id: str
    prompt_format: str
    model_type: ModelType
    
@dataclasses.dataclass
class ModelConfig:
    id: str
    prompt_format: str
    model_type: ModelType

SETTINGS: List[str] = ["sep", "joint"]
    
TASK_MAPPING: Dict[str, TaskConfig] = {
    "TACRED": TaskConfig(
        id="TACRED",
        max_tokens=512,
        sep=SepConfig(
            max_tokens_extract=300,
            max_tokens_refine=300,
            max_tokens_label_mapping=300,
            max_tokens_confidence=500,
            max_tokens_relation_extraction=100,
            extract_prompt=EXTRACT_ENTITY_TYPE_PROMPT,
            refine_prompt=REFINEMENT_LABELS_PROMPT,
            confidence_prompt=CONFIDENCE_SCORE_PROMPT,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT,
        ),
        joint = None,
    ),
    "FewRel": TaskConfig(
        id="FewRel",
        max_tokens=512,
        sep=SepConfig(
            max_tokens_extract=300,
            max_tokens_refine=400,
            max_tokens_label_mapping=400,
            max_tokens_confidence=400,
            max_tokens_relation_extraction=150,
            extract_prompt=EXTRACT_ENTITY_TYPE_PROMPT,
            refine_prompt=REFINEMENT_LABELS_PROMPT_FEWREL,
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_FEWREL_SEP,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_FEWREL,
        ),
        joint = JointConfig(
            max_tokens_extract_refine=300,
            max_tokens_confidence=300,
            max_tokens_relation_extraction=100,
            extract_refine_prompt=EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_FEWREL_PROMPT,
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_FEWREL_JOINT,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_FEWREL,
        )
    ),
    "wiki": TaskConfig(
        id="wiki",
        max_tokens=512,
        sep=SepConfig(
            max_tokens_extract=150,
            max_tokens_refine=300,
            max_tokens_label_mapping=300,
            max_tokens_confidence=300,
            max_tokens_relation_extraction=100,
            extract_prompt=EXTRACT_ENTITY_TYPE_PROMPT_WIKI, 
            refine_prompt=REFINEMENT_LABELS_PROMPT_WIKI,  
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_WIKI_SEP,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_WIKI,
        ),
        joint = JointConfig(
            max_tokens_extract_refine=300,
            max_tokens_confidence=300,
            max_tokens_relation_extraction=100,
            extract_refine_prompt=EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_WIKI_PROMPT,
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_WIKI_JOINT,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_WIKI,
    ),
    ),
}

MISTRAL_PROMPT_FORMAT = (
    "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n {command}\n"
)
LLAMA_PROMPT_FORMAT = (
    """<s>[INST] <<SYS>>{prompt}\n<</SYS>>\n{command} [/INST]"""
)

MODEL_MAPPING: Dict[str, ModelConfig] ={
    "llama2_13b": ModelConfig(
        id="meta-llama/Llama-2-13b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        model_type=ModelType.PROTECTED,
    ),
    
    "llama2_70b": ModelConfig(
        id="meta-llama/Llama-2-70b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        model_type=ModelType.PROTECTED,
    ),
    
    "mistral": ModelConfig(
        id="mistralai/Mistral-7B-Instruct-v0.1",
        prompt_format=MISTRAL_PROMPT_FORMAT,
        model_type=ModelType.PROTECTED,
    ),
    
    "zephyr": ModelConfig(
        id="HuggingFaceH4/zephyr-7b-beta",
        prompt_format=MISTRAL_PROMPT_FORMAT,
        model_type=ModelType.PROTECTED,
    ),
    
    "gpt": ModelConfig(
        id = "gpt-3.5-turbo",
        prompt_format = None,
        model_type = ModelType.GPT,
    ),
    
    "mixtral": ModelConfig(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt_format=None,
        model_type=ModelType.OPEN,
    ),
}

def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def import_model_and_tokenizer(model: ModelConfig, access_token: str = None):
    if model.model_type == ModelType.PROTECTED:
        if not access_token:
            access_token = os.getenv("HF_ACCESS_TOKEN")
            if not access_token:
                raise ValueError("You must provide an access token to import a protected model.")
        login(token=access_token)
        
    bnb_config = get_bnb_config()
    
    model_kwargs = {
        "quantization_config": bnb_config,
        "use_cache": True,
        "device_map": "auto",
    }
    
    if model.model_type == ModelType.PROTECTED:
        model_kwargs["token"] = access_token   
    
    language_model = AutoModelForCausalLM.from_pretrained(model.id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return language_model, tokenizer
    
    