import json 
import sys
from typing import Dict
import openai

from src.labels import (
    TACRED_LABELS, 
    TACREV_LABELS
)


from src.utils import (
    TaskConfig,
    import_model_and_tokenizer,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)


class ChainofRefinement:
    def __init__(
        self, model_id, top_p, temperature, task, setting, data, access_token
    ):
        self.model_id = model_id
        self.model_config: ModelConfig = MODEL_MAPPING.get(model_id, None)
        if self.model_config is None:
            print(f"Invalid model. Valid models are: {', '.join(MODEL_MAPPING.keys())}")
            sys.exit()
        self.task = task
        self.task_config: TaskConfig = TASK_MAPPING.get(task, None)
        if self.task_config is None:
            print(f"Invalid task. Valid taks are: {', '.join(TASK_MAPPING.keys())}")
            sys.exit()
        self.setting = setting
        if self.setting not in SETTINGS:
            print(f"Invalid setting. Valid settings are: {', '.join(SETTINGS)}")
            sys.exit()
        if self.task_config.__dict__[self.setting] is None:
            print(
                f"Invalid combination. Settings {self.setting} was not implemented for task {self.task}"
            )
            sys.exit()
            
        #self.labels = labels
        self.labels = self.task_config.sep.labels
        self.access_token = access_token
        self.data = data
        self.top_p = top_p
        self.temperature = temperature
        
        if not self.model_config.is_gpt:
            self.model, self.tokenizer = import_model_and_tokenizer(
                self.model_config, access_token=self.access_token
            )
        else:
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int) -> str:
        if self.model_config.is_protected:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            tokens = self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0][0:]
            tokens = tokens.split("[/INST]")[1]
            
            if len(tokens.split("\n\n")) > 1 and tokens.split("\n\n")[1] is not None:
                tokens = tokens.split("\n\n")[1]
            else:
                tokens = tokens
            return tokens
        else:
            print("Not implemented yet")
        #TODO: implement for non-protected models
        
    def generate_gpt_response(self, prompt: str, max_tokens: int) -> str:
        if self.model_config.is_gpt:
            response = openai.ChatCompletion.create(
                model=self.model_config.id,
                messages=[
                {"role": "system", "content": "You are a laconic assistant in information extraction. You reply with brief, to-the-point answers."},
                {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                )
            return response['choices'][0]['message']['content']
        else:
            print("The model is not a GPT model. Please use a GPT model.")
        
    def process_record(self, record):
            sentence = record['token']
            sentence = " ".join(sentence)
            head_entity = record['h']['name']
            tail_entity = record['t']['name']
            relation = record['relation']
            return sentence, head_entity, tail_entity, relation
    
    def extract_refinement_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_prompt = self.task_config.sep.extract_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity
        )
        extract_prompt = self.model_config.prompt_format.format(
            prompt=extract_prompt, command=self.task_config.sep.extract_command
        )
        extract_response = self.generate_response(
            extract_prompt, self.task_config.sep.max_tokens_extract
        )
        #print(f"""extract_response: {extract_response}""")
        refine_prompt = self.task_config.sep.refine_prompt.format(
            sentence=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            relation_labels=self.labels,
        )
        refine_prompt = self.model_config.prompt_format.format(
            prompt=refine_prompt, command=self.task_config.sep.refine_command
        )
        refine_response = self.generate_response(
            refine_prompt, self.task_config.sep.max_tokens_refine
        )
        label_mapping_prompt = self.task_config.sep.label_mapping_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, refined_relation_labels=refine_response, relation_labels=self.labels
        )
        label_mapping_prompt = self.model_config.prompt_format.format(
            prompt=label_mapping_prompt, command=self.task_config.sep.label_mapping_command
        )
        label_mapping_response = self.generate_response(
            label_mapping_prompt, self.task_config.sep.max_tokens_label_mapping
        )
        
        confidence_prompt = self.task_config.sep.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            refined_relation_labels=label_mapping_response,
        )
        confidence_prompt = self.model_config.prompt_format.format(
            prompt=confidence_prompt, command=self.task_config.sep.confidence_command
        )
        confidence_response = self.generate_response(
            confidence_prompt, self.task_config.sep.max_tokens_confidence
        )
        #print(f"""confidence_response: {confidence_response}""")
        relation_extraction_prompt = self.task_config.sep.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=label_mapping_response,
            relationship_confidence_scores=confidence_response,
        )
        relation_extraction_prompt = self.model_config.prompt_format.format(
            prompt=relation_extraction_prompt,
            command=self.task_config.sep.relation_extraction_command,
        )
        relation_extraction_response = self.generate_response(
            relation_extraction_prompt,
            self.task_config.sep.max_tokens_relation_extraction,
        )
        return extract_response, refine_response, label_mapping_response, confidence_response, relation_extraction_response
    
    def extract_gpt_refinement_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_prompt = self.task_config.sep.extract_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity
        )
        extract_prompt = extract_prompt + self.task_config.sep.extract_command
        extract_response = self.generate_gpt_response(
            extract_prompt, self.task_config.sep.max_tokens_extract
        )
        refine_prompt = self.task_config.sep.refine_prompt.format(
            sentence=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            relation_labels=self.labels,
        )
        refine_prompt = refine_prompt + self.task_config.sep.refine_command
        refine_response = self.generate_gpt_response(
            refine_prompt, self.task_config.sep.max_tokens_refine
        )
        confidence_prompt = self.task_config.sep.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            refined_relation_labels=refine_response,
        )
        confidence_prompt = confidence_prompt + self.task_config.sep.confidence_command
        confidence_response = self.generate_gpt_response(
            confidence_prompt, self.task_config.sep.max_tokens_confidence
        )
        relation_extraction_prompt = self.task_config.sep.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=refine_response,
            relationship_confidence_scores=confidence_response,
        )
        relation_extraction_prompt = relation_extraction_prompt + self.task_config.sep.relation_extraction_command
        relation_extraction_response = self.generate_gpt_response(
            relation_extraction_prompt,
            self.task_config.sep.max_tokens_relation_extraction,
        )
        
        return extract_response, refine_response, confidence_response, relation_extraction_response
        #print(f"""extract_response: {extract_response}""")
            
    
    def print_results(self, result: Dict[str, str]):
        for k, v in result.items():
            print(f"{k}: {v}")
            print("--------------------\n")
        print("=====================================\n")
        
    def run_chain(self):
        all_results = []
        for rec in self.data:
            if not self.model_config.is_gpt:
                sentence, head_entity, tail_entity, relation = self.process_record(rec)
                # print(f"""sentence: {sentence}""")
                # print(f"""head_entity: {head_entity}""")
                # print(f"""tail_entity: {tail_entity}""")
                if self.setting == "sep":
                    extract_response, refine_response, label_mapping_response, confidence_response, relation_extraction_response = self.extract_refinement_chain(
                        sentence, head_entity, tail_entity
                    )
                    result = {
                        "sentence": sentence,
                        "head_entity": head_entity,
                        "tail_entity": tail_entity,
                        "relation": relation,
                        "extract_response": extract_response,
                        "refine_response": refine_response,
                        "label_mapping_response": label_mapping_response,
                        "confidence_response": confidence_response,
                        "relation_extraction_response": relation_extraction_response,
                    }
                    all_results.append(result)
                    self.print_results(result)
            elif self.model_config.is_gpt:
                sentence, head_entity, tail_entity, relation = self.process_record(rec)
                #print(f"""sentence: {sentence}""")
                if self.setting == "sep":
                    extract_response, refine_response, confidence_response, relation_extraction_response = self.extract_gpt_refinement_chain(
                        sentence, head_entity, tail_entity
                    )
                    #print(f"""extract_response: {extract_response}""")
                    result = {
                        "sentence": sentence,
                        "head_entity": head_entity,
                        "tail_entity": tail_entity,
                        "relation": relation,
                        "extract_response": extract_response,
                        "refine_response": refine_response,
                        "confidence_response": confidence_response,
                        "relation_extraction_response": relation_extraction_response,
                    }
            else:
                print("Not implemented yet")
                #TODO: implement for joint chain
        
        result_file_path = f"results/{self.model_id}_{self.task}_{self.setting}_test.json"
        with open(result_file_path, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)
        
    
    