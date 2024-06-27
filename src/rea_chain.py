import json 
from typing import List, Dict, Tuple
import openai
from enum import Enum
import logging
from trl import setup_chat_format
from dataclasses import dataclass
from src.utils import SepConfig, JointConfig, ModelType, SETTINGS, TASK_MAPPING, MODEL_MAPPING

logging.basicConfig(filename='chain_of_refinement.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    PROTECTED = "protected"
    OPEN = "open"
    GPT = "gpt"

class CostumException(Exception):
    pass

from src.utils import (
    TaskConfig,
    import_model_and_tokenizer,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)

@dataclass
class ChainConfig:
    model_id: str
    top_p: float
    temperature: float
    task: str
    setting: str
    labels: List[str]

class ReaChain:
    def __init__(
        self, config: ChainConfig, data: List[Dict], access_token: str
    ):
        self.config = config
        self.model_config = MODEL_MAPPING.get(config.model_id, None)
        self.task_config = TASK_MAPPING.get(config.task, None)
        self.data = data
        self.access_token = access_token
        
        self._validate_config()
        self._initialize_model()
        
        logger.info("Initialized ReaChain with model_id: {config.model_id}, task: {config.task}, setting: {config.setting}")
        
    def _validate_config(self):
        if self.model_config is None:
            raise CostumException(f"Invalid model. Valid models are: {', '.join(MODEL_MAPPING.keys())}")
        if self.task_config is None:
            raise CostumException(f"Invalid task. Valid taks are: {', '.join(TASK_MAPPING.keys())}")
        if self.config.setting not in SETTINGS:
            raise CostumException(f"Invalid setting. Valid settings are: {', '.join(SETTINGS)}")
        if getattr(self.task_config, self.config.setting) is None:
            raise CostumException(
                f"Invalid combination. Settings {self.config.setting} was not implemented for task {self.config.task}"
            )
    
    def _initialize_model(self):
        if self.model_config.model_type != ModelType.GPT:
            self.model, self.tokenizer = import_model_and_tokenizer(
                self.model_config, access_token=self.access_token
            )
        else:
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int) -> str:
        if self.model_config.model_type == ModelType.PROTECTED:
            return self._generate_protected_response(prompt, max_tokens)
        elif self.model_config.model_type == ModelType.OPEN:
            return self._generate_open_response(prompt, max_tokens)
        elif self.model_config.model_type == ModelType.GPT:
            return self._generate_gpt_response(prompt, max_tokens)
        else:
            raise CostumException(f"Model type {self.model_config.model_type} is not implemented yet")
    
    def _generate_protected_response(self, prompt: str, max_tokens: int) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
        )
        tokens = self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][0:]
        
        return self._process_tokens(tokens)
        
    def _generate_open_response(self, prompt: str, max_tokens: int) -> str:
        messages = [
                {"role": "user", "content": "You are a helpful assistant in information extraction task. You reply with brief, to-the-point answers. Provide explanations for your answers."},
                {"role": "assistant", "content": "Perfect, I am ready to help you for the information extraction task."},
                {"role": "user", "content": prompt},
            ]
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id # Maybe this is not needed
        )
        tokens = self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][0:]
        
        return self._process_tokens(tokens)
    
    def _generate_gpt_response(self, prompt: str, max_tokens: int) -> str:
        response = openai.ChatCompletion.create(
                model=self.model_config.id,
                messages=[
                {"role": "system", "content": "You are a helpful assistant in information extraction task. You reply with brief, to-the-point answers. Provide explanations for your answers."},
                {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                )
            
        return response['choices'][0]['message']['content']
            
    def _get_gpt_asnwer(self, prompt: str, max_tokens: int, engine: str = "gpt-3.5-turbo-instruct") -> str:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            )
        return response['choices'][0]['text'].strip()
    
    def _process_tokens(self, tokens: str) -> str:
        tokens = tokens.split("[/INST]")[1] if "[/INST]" in tokens else tokens
        return tokens.split("\n\n") if len(tokens.split("\n\n")) > 1 and tokens.split("\n\n")[1] is not None else tokens
    
    def process_record(self, record: Dict) -> Tuple[str, str, str, str]:
        if self.task_config.id != "wiki":
            return self._process_standard_record(record)
        else:
            return self._process_wiki_record(record)
    
    def _process_standard_record(self, record: Dict) -> Tuple[str, str, str, str]:
        sentence = " ".join(record['token'])
        return sentence, record['h']['name'], record['t']['name'], record['relation']
    
    def _process_wiki_record(self, record: Dict) -> Tuple[str, str, str, str]:
        sentence = record['sentence']
        relations = record['relations']
        for relation in relations:
            head_entity = relation['head']['name']
            tail_entity = relation['tail']['name']
            relation_type = relation['type']
        return sentence, head_entity, tail_entity, relation_type
    
    def run_chain(self):
        all_results = []
        result_file_path = f"results/{self.config.model_id}_{self.config.task}_{self.config.setting}_test.json"
        
        for i, rec in enumerate(self.data):
            sentence, head_entity, tail_entity, relation = self.process_record(rec)
            result = self._process_chain(sentence, head_entity, tail_entity, relation)
            all_results.append(result)
            
            if (i + 1) % 10 == 0:
                self._save_results(all_results, result_file_path)
                
            self._print_results(result)
        
        self._save_results(all_results, result_file_path)
    
    def _process_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str) -> Dict[str, str]:
        if self.config.setting == "sep":
            return self._process_sep_chain(sentence, head_entity, tail_entity, relation)
        elif self.config.setting == "joint":
            return self._process_joint_chain(sentence, head_entity, tail_entity, relation)
        else:
            raise CostumException(f"Setting {self.config.setting} is not implemented yet")
    
    def _process_sep_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str) -> Dict[str, str]:
        sep_config = self.task_config.sep
        
        if self.model_config.model_type == ModelType.PROTECTED:
            return self._process_protected_sep_chain(sentence, head_entity, tail_entity, relation, sep_config)
        elif self.model_config.model_type == ModelType.OPEN:
            return self._process_open_sep_chain(sentence, head_entity, tail_entity, relation, sep_config)
        elif self.model_config.model_type == ModelType.GPT:
            return self._process_gpt_sep_chain(sentence, head_entity, tail_entity, relation, sep_config)
        else:
            raise CostumException(f"Model type {self.model_config.model_type} is not implemented yet")
        
    def _process_protected_sep_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, sep_config: SepConfig) -> Dict[str, str]:
        # Implementations for protected models (e.g., Llama 2)
        extract_prompt = self._format_prompt(sep_config.extract_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity)
        extract_prompt = self.model_config.prompt_format.format(prompt=extract_prompt, command=sep_config.extract_command)
        # Extract Entity Types
        extract_response = self.generate_response(extract_prompt, sep_config.max_tokens_extract)
        
        refine_prompt = self._format_prompt(sep_config.refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, relation_labels=self.config.labels)
        refine_prompt = self.model_config.prompt_format.format(prompt=refine_prompt, command=sep_config.refine_command)
        # Refine Relation Labels
        refine_response = self.generate_response(refine_prompt, sep_config.max_tokens_refine)
        
        confidence_prompt = self._format_prompt(sep_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, refined_relation_labels=refine_response)
        confidence_prompt = self.model_config.prompt_format.format(prompt=confidence_prompt, command=sep_config.confidence_command)
        # Relationship Confidence Scores
        confidence_response = self.generate_response(confidence_prompt, sep_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(sep_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt = self.model_config.prompt_format.format(prompt=relation_extraction_prompt, command=sep_config.relation_extraction_command)
        # Relation Extraction
        relation_extraction_response = self.generate_response(relation_extraction_prompt, sep_config.max_tokens_relation_extraction)
        
        return self._create_sep_result_dict(sentence, head_entity, tail_entity, relation, extract_response, refine_response, confidence_response, relation_extraction_response)
    
    def _process_open_sep_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, sep_config: SepConfig) -> Dict[str, str]:
        # Implementations for open models (e.g., Mixtral)
        extract_prompt = self._format_prompt(sep_config.extract_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity)
        extract_prompt += sep_config.extract_command
        # Extract Entity Types
        extract_response = self.generate_response(extract_prompt, sep_config.max_tokens_extract)
        
        refine_prompt = self._format_prompt(sep_config.refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, relation_labels=self.config.labels)
        refine_prompt += sep_config.refine_command
        # Refine Relation Labels
        refine_response = self.generate_response(refine_prompt, sep_config.max_tokens_refine)
        
        confidence_prompt = self._format_prompt(sep_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, refined_relation_labels=refine_response)
        confidence_prompt += sep_config.confidence_command
        # Relationship Confidence Scores
        confidence_response = self.generate_response(confidence_prompt, sep_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(sep_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt += sep_config.relation_extraction_command
        # Relation Extraction
        relation_extraction_response = self.generate_response(relation_extraction_prompt, sep_config.max_tokens_relation_extraction)
        
        return self._create_sep_result_dict(sentence, head_entity, tail_entity, relation, extract_response, refine_response, confidence_response, relation_extraction_response)
    
    def _process_gpt_sep_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, sep_config: SepConfig) -> Dict[str, str]:
        # Implementations for GPT models
        extract_prompt = self._format_prompt(sep_config.extract_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity)
        extract_prompt += sep_config.extract_command
        # Extract Entity Types
        extract_response = self.generate_gpt_response(extract_prompt, sep_config.max_tokens_extract)
        
        refine_prompt = self._format_prompt(sep_config.refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, relation_labels=self.config.labels)
        refine_prompt += sep_config.refine_command
        # Refine Relation Labels
        refine_response = self.generate_gpt_response(refine_prompt, sep_config.max_tokens_refine)
        
        confidence_prompt = self._format_prompt(sep_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_response, refined_relation_labels=refine_response)
        confidence_prompt += sep_config.confidence_command
        # Relationship Confidence Scores
        confidence_response = self.generate_gpt_response(confidence_prompt, sep_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(sep_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt += sep_config.relation_extraction_command
        # Relation Extraction
        relation_extraction_response = self.generate_gpt_response(relation_extraction_prompt, sep_config.max_tokens_relation_extraction)
        
        return self._create_sep_result_dict(sentence, head_entity, tail_entity, relation, extract_response, refine_response, confidence_response, relation_extraction_response)
    
    def _process_joint_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str) -> Dict[str, str]:
        joint_config = self.task_config.joint
        
        if self.model_config.model_type == ModelType.PROTECTED:
            return self._process_protected_joint_chain(sentence, head_entity, tail_entity, relation, joint_config)
        elif self.model_config.model_type == ModelType.OPEN:
            return self._process_open_joint_chain(sentence, head_entity, tail_entity, relation, joint_config)
        elif self.model_config.model_type == ModelType.GPT:
            return self._process_gpt_joint_chain(sentence, head_entity, tail_entity, relation, joint_config)
        else:
            raise CostumException(f"Model type {self.model_config.model_type} is not implemented yet")
        
    def _process_protected_joint_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, joint_config: JointConfig) -> Dict[str, str]:
        # Implementations for protected models (e.g., Llama 2)
        extract_refine_prompt = self._format_prompt(joint_config.extract_refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, relation_labels=self.config.labels)
        extract_refine_prompt = self.model_config.prompt_format.format(prompt=extract_refine_prompt, command=joint_config.extract_refine_command)
        # Extract and Refine Relation Labels
        extract_refine_response = self.generate_response(extract_refine_prompt, joint_config.max_tokens_extract_refine)
        
        confidence_prompt = self._format_prompt(joint_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_refine_response)
        confidence_prompt = self.model_config.prompt_format.format(prompt=confidence_prompt, command=joint_config.confidence_command)
        # Relationship Confidence Scores
        confidence_response = self.generate_response(confidence_prompt, joint_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(joint_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=extract_refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt = self.model_config.prompt_format.format(prompt=relation_extraction_prompt, command=joint_config.relation_extraction_command)
        # Relation Extraction
        relation_extraction_response = self.generate_response(relation_extraction_prompt, joint_config.max_tokens_relation_extraction)
        
        return self._create_joint_result_dict(sentence, head_entity, tail_entity, relation, extract_refine_response, confidence_response, relation_extraction_response)
    
    def _process_open_joint_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, joint_config: JointConfig) -> Dict[str, str]:
        # Implementations for open models (e.g., Mixtral)
        extract_refine_prompt = self._format_prompt(joint_config.extract_refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, relation_labels=self.config.labels)
        extract_refine_prompt += joint_config.extract_refine_command
        # Extract and Refine Relation Labels
        extract_refine_response = self.generate_response(extract_refine_prompt, joint_config.max_tokens_extract_refine)
        
        confidence_prompt = self._format_prompt(joint_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_refine_response)
        confidence_prompt += joint_config.confidence_command
        # Relationship Confidence Scores
        confidence_response = self.generate_response(confidence_prompt, joint_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(joint_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=extract_refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt += joint_config.relation_extraction_command
        # Relation Extraction
        relation_extraction_response = self.generate_response(relation_extraction_prompt, joint_config.max_tokens_relation_extraction)
        
        return self._create_joint_result_dict(sentence, head_entity, tail_entity, relation, extract_refine_response, confidence_response, relation_extraction_response)
    
    def _process_gpt_joint_chain(self, sentence: str, head_entity: str, tail_entity: str, relation: str, joint_config: JointConfig) -> Dict[str, str]:
        # Implementations for GPT models
        extract_refine_prompt = self._format_prompt(joint_config.extract_refine_prompt, sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, relation_labels=self.config.labels)
        extract_refine_prompt += joint_config.extract_refine_command
        # Extract and Refine Relation Labels
        extract_refine_response = self.generate_gpt_response(extract_refine_prompt, joint_config.max_tokens_extract_refine)
        
        confidence_prompt = self._format_prompt(joint_config.confidence_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, entity_types=extract_refine_response)
        confidence_prompt += joint_config.confidence_command
        # Relationship Confidence Scores
        confidence_response = self.generate_gpt_response(confidence_prompt, joint_config.max_tokens_confidence)
        
        relation_extraction_prompt = self._format_prompt(joint_config.relation_extraction_prompt, input_text=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=extract_refine_response, relationship_confidence_scores=confidence_response)
        relation_extraction_prompt += joint_config.relation_extraction_command
        # Relation Extraction
        relation_extraction_response = self.generate_gpt_response(relation_extraction_prompt, joint_config.max_tokens_relation_extraction)
        
        return self._create_joint_result_dict(sentence, head_entity, tail_entity, relation, extract_refine_response, confidence_response, relation_extraction_response)
    
    def _format_prompt(self, prompt: str, **kwargs) -> str:
        return prompt.format(**kwargs)
    
    def _create_sep_result_dict(self, sentence: str, head_entity: str, tail_entity: str, relation: str, extract_response: str, refine_response: str, confidence_response: str, relation_extraction_response: str) -> Dict[str, str]:
        result = {
            "sentence": sentence,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "relation": relation,
            "extract_response": extract_response,
            "refine_response": refine_response,
            "confidence_response": confidence_response,
            "relation_extraction_response": relation_extraction_response
        }
        logger.info(f"Processed record: {result}")
        return result
    
    def _create_joint_result_dict(self, sentence: str, head_entity: str, tail_entity: str, relation: str, extract_refine_response: str, confidence_response: str, relation_extraction_response: str) -> Dict[str, str]:
        result = {
            "sentence": sentence,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
            "relation": relation,
            "extract_refine_response": extract_refine_response,
            "confidence_response": confidence_response,
            "relation_extraction_response": relation_extraction_response
        }
        logger.info(f"Processed record: {result}")
        return result
    
        
    def _save_results(self, results: List[Dict], file_path: str):
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)

    def _print_results(self, result: Dict[str, str]):
        for k, v in result.items():
            logger.info(f"{k}: {v}")
        logger.info("=" * 40)
        