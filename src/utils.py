import dataclasses
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

from src.prompts import (
    EXTRACT_ENTITY_TYPE_PROMPT,
    REFINEMENT_LABELS_PROMPT,
    CONFIDENCE_SCORE_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    CONFIDENCE_SCORE_PROMPT,
    REFINEMENT_LABELS_PROMPT_FEWREL,
    CONFIDENCE_SCORE_PROMPT_FEWREL,
    RELATION_EXTRACTION_PROMPT_FEWREL,
    EXTRACT_ENTITY_TYPE_PROMPT_WIKI,
    REFINEMENT_LABELS_PROMPT_WIKI,
    CONFIDENCE_SCORE_PROMPT_WIKI,
    CONFIDENCE_SCORE_PROMPT_WIKI_JOINT,
    RELATION_EXTRACTION_PROMPT_WIKI,
    EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_PROMPT_WIKI,
    EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_FEWREL_PROMPT,
    CONFIDENCE_SCORE_PROMPT_FEWREL_JOINT,
    
)
from src.labels import TACRED_LABELS

SETTINGS = ["sep", "joint"]

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
    
    
TASK_MAPPING = {
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
            extract_prompt=EXTRACT_ENTITY_TYPE_PROMPT, #TODO change this
            refine_prompt=REFINEMENT_LABELS_PROMPT_FEWREL,  #TODO change this
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_FEWREL, #TODO change this
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_FEWREL, #TODO change this
            #labels=None,
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
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_WIKI, 
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_WIKI, 
            #labels=None,
        ),
        joint = JointConfig(
            max_tokens_extract_refine=300,
            max_tokens_confidence=300,
            max_tokens_relation_extraction=100,
            extract_refine_prompt=EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_PROMPT_WIKI,
            confidence_prompt=CONFIDENCE_SCORE_PROMPT_WIKI_JOINT,
            relation_extraction_prompt=RELATION_EXTRACTION_PROMPT_WIKI,
    ),
    ),
}

@dataclasses.dataclass
class ModelConfig:
    id: str
    prompt_format: str
    is_protected: bool
    is_gpt: bool
    
MISTRAL_PROMPT_FORMAT = (
    "<|system|>\n</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n {command}\n"
)
LLAMA_PROMPT_FORMAT = (
    """<s>[INST] <<SYS>>{prompt}\n<</SYS>>\n{command} [/INST]"""
)

MODEL_MAPPING = {
    "llama2_13b": ModelConfig(
        id="meta-llama/Llama-2-13b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        is_protected=True,
        is_gpt=False,
    ),
    "llama2_70b": ModelConfig(
        id="meta-llama/Llama-2-70b-chat-hf",
        prompt_format=LLAMA_PROMPT_FORMAT,
        is_protected=True,
        is_gpt=False,
    ),
    "mistral": ModelConfig(
        id="mistralai/Mistral-7B-Instruct-v0.1",
        prompt_format=MISTRAL_PROMPT_FORMAT,
        is_protected=False,
        is_gpt=False,
    ),
    "zephyr": ModelConfig(
        id="HuggingFaceH4/zephyr-7b-beta",
        prompt_format=MISTRAL_PROMPT_FORMAT,
        is_protected=False,
        is_gpt=False,
    ),
    "gpt": ModelConfig(
        id = "gpt-3.5-turbo",
        prompt_format = None,
        is_protected = False,
        is_gpt = True,
    ),
    "mixtral": ModelConfig(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt_format=None,
        is_protected=False,
        is_gpt=False,
    ),
    "Llama_3": ModelConfig(
        id = "meta-llama/Meta-Llama-3-8B",
        prompt_format=None,
        is_protected=False,
        is_gpt=False,
    ),
}

def import_model_and_tokenizer(model: ModelConfig, access_token: str = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    if model.is_protected:
        if access_token is None:
            raise ValueError(
                "You must provide an access token to import a protected model."
            )
        elif access_token is not None:
            login(token=access_token)

            language_model = AutoModelForCausalLM.from_pretrained(model.id,
                                                                  quantization_config=bnb_config,
                                                                  use_cache=True,
                                                                  device_map="auto",
                                                                  token=access_token)
            
        elif model.is_protected==False and model.is_gpt==False: #MIXTRAL
            language_model = AutoModelForCausalLM.from_pretrained(model.id,
                                                                  #load_in_4bit=True,
            )
    else:
        language_model = AutoModelForCausalLM.from_pretrained(model.id,
                                                              quantization_config=bnb_config,
                                                              use_cache=True,
                                                              device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model.id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return language_model, tokenizer
    
    