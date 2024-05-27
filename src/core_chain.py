import json 
import sys
from typing import Dict
import openai
import logging
from trl import setup_chat_format

logging.basicConfig(filename='chain_of_refinement.log', level=logging.INFO)

from src.labels import TACRED_LABELS

from src.utils import (
    TaskConfig,
    import_model_and_tokenizer,
    MODEL_MAPPING,
    ModelConfig,
    TASK_MAPPING,
    SETTINGS,
)


class ReaChain:
    def __init__(
        self, model_id, top_p, temperature, task, setting, data, access_token, labels
    ):
        self.model_id = model_id
        self.model_config: ModelConfig = MODEL_MAPPING.get(model_id, None)
        if self.model_config is None:
            print(f"Invalid model. Valid models are: {', '.join(MODEL_MAPPING.keys())}")
            sys.exit()
        self.task = task
        self.task_config: TaskConfig = TASK_MAPPING.get(task, None)
        print(f"""task_config: {self.task_config}""")
        print("\n\n")
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
            
        logging.info(
            f"Initialized ReaChain with model_id: {self.model_id}, task: {self.task}, setting: {self.setting}"
                     )
        
        self.labels = labels
        #self.labels = self.task_config.sep.labels
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
            # messages= [
            #     {"role": "user", "content": "You are a helpful assistant in information extraction task. You reply with brief, to-the-point answers. Provide explanations for your answers."},
            #     {"role": "assistant", "content": "Perfect, I am ready to help you for the information extraction task."},
            #     {"role": "user", "content": prompt},
            # ]
            # input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
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
            
            logging.info(f"Generated response for prompt:{prompt} is: {tokens}")
            
            return tokens
        elif self.model_config.is_gpt==False and self.model_config.is_protected==False: #MIXTRAL
            messages = [
                {"role": "user", "content": "You are a helpful assistant in information extraction task. You reply with brief, to-the-point answers. Provide explanations for your answers."},
                {"role": "assistant", "content": "Perfect, I am ready to help you for the information extraction task."},
                {"role": "user", "content": prompt},
            ]
            self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
            input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
            
            # input_ids = self.tokenizer(
            #     prompt, return_tensors="pt", truncation=True
            # ).input_ids.cuda()
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
            tokens = self.tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0][len(prompt)-1:]
            tokens = tokens.split("[/INST]")[1] if "[/INST]" in tokens else tokens
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
                {"role": "system", "content": "You are a helpful assistant in information extraction task. You reply with brief, to-the-point answers. Provide explanations for your answers."},
                {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens,
                )
            
            #logging.info(
            #    f"Generated response for prompt:{prompt} is: {response['choices'][0]['message']['content']}"
            #    )
            
            return response['choices'][0]['message']['content']
        else:
            print("The model is not a GPT model. Please use a GPT model.")
            
    def get_gpt_asnwer(self, prompt: str, max_tokens: int, engine: str = "gpt-3.5-turbo-instruct"):
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
        
    def process_record(self, record):
            sentence = record['token']
            sentence = " ".join(sentence)
            head_entity = record['h']['name']
            tail_entity = record['t']['name']
            relation = record['relation']
            return sentence, head_entity, tail_entity, relation
        
    def process_wiki_record(self, record):
        sentence = record['sentence']
        relations = record['relations']
        for relation in relations:
            head_entity = relation['head']['name']
            tail_entity = relation['tail']['name']
            relation_type = relation['type']
        return sentence, head_entity, tail_entity, relation_type
    
    def extract_mixtral_sep_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_prompt = self.task_config.sep.extract_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity
        )
        extract_prompt = extract_prompt + self.task_config.sep.extract_command
        extract_response = self.generate_response(
            extract_prompt, self.task_config.sep.max_tokens_extract
        )
        print("extract_response: ", extract_response)
        
        refine_prompt = self.task_config.sep.refine_prompt.format(
            sentence=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            relation_labels=self.labels,
        )
        refine_prompt = refine_prompt + self.task_config.sep.refine_command
        refine_response = self.generate_response(
            refine_prompt, self.task_config.sep.max_tokens_refine
        )
        print("refine_response: ", refine_response)
        
        confidence_prompt = self.task_config.sep.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            refined_relation_labels=refine_response,
        )
        confidence_prompt = confidence_prompt + self.task_config.sep.confidence_command
        confidence_response = self.generate_response(
            confidence_prompt, self.task_config.sep.max_tokens_confidence
        )
        print("confidence_response: ", confidence_response)
        
        relation_extraction_prompt = self.task_config.sep.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=refine_response,
            relationship_confidence_scores=confidence_response,
        )
        relation_extraction_prompt = relation_extraction_prompt + self.task_config.sep.relation_extraction_command
        relation_extraction_response = self.generate_response(
            relation_extraction_prompt,
            self.task_config.sep.max_tokens_relation_extraction,
        )
        print("relation_extraction_response: ", relation_extraction_response)
        return extract_response, refine_response, confidence_response, relation_extraction_response
            
    def extract_llama_sep_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_prompt = self.task_config.sep.extract_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity
        )
        extract_prompt = self.model_config.prompt_format.format(
            prompt=extract_prompt, command=self.task_config.sep.extract_command
        )
        extract_response = self.generate_response(
            extract_prompt, self.task_config.sep.max_tokens_extract
        )
        
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
        
        confidence_prompt = self.task_config.sep.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_response,
            refined_relation_labels=refine_response,
        )
        confidence_prompt = self.model_config.prompt_format.format(
            prompt=confidence_prompt, command=self.task_config.sep.confidence_command
        )
        confidence_response = self.generate_response(
            confidence_prompt, self.task_config.sep.max_tokens_confidence
        )
        relation_extraction_prompt = self.task_config.sep.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=refine_response,
            relationship_confidence_scores=confidence_response,
        )
        relation_extraction_prompt = self.model_config.prompt_format.format(
            prompt=relation_extraction_prompt,
            command=self.task_config.sep.relation_extraction_command,
        )
        # relation_extraction_prompt = relation_extraction_prompt + "\n" + self.task_config.sep.relation_extraction_command
        relation_extraction_response = self.generate_response(
            relation_extraction_prompt,
            self.task_config.sep.max_tokens_relation_extraction,
        )
        return extract_response, refine_response, confidence_response, relation_extraction_response
    
    def extract_llama_joint_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_refine_prompt = self.task_config.joint.extract_refine_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, relation_labels=self.labels
        )
        extract_refine_prompt = self.model_config.prompt_format.format(
            prompt=extract_refine_prompt, command=self.task_config.joint.extract_refine_command
        )
        extract_refine_response = self.generate_response(
            extract_refine_prompt, self.task_config.joint.max_tokens_extract_refine
        )
        
        confidence_prompt = self.task_config.joint.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_refine_response,
        )
        confidence_prompt = self.model_config.prompt_format.format(
            prompt=confidence_prompt, command=self.task_config.joint.confidence_command
        )
        confidence_response = self.generate_response(
            confidence_prompt, self.task_config.joint.max_tokens_confidence
        )
        
        relation_extraction_prompt = self.task_config.joint.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=extract_refine_response,
            relationship_confidence_scores=confidence_response,
        )   
        relation_extraction_prompt = self.model_config.prompt_format.format(
            prompt=relation_extraction_prompt, command=self.task_config.joint.relation_extraction_command
        )
        relation_extraction_response = self.generate_response(
            relation_extraction_prompt,
            self.task_config.joint.max_tokens_relation_extraction,
        )
        return extract_refine_response, confidence_response, relation_extraction_response
    
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
        print(f"""refine_prompt: {refine_prompt}""")
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
        logging.info(
            f"the sentence is: {sentence}, and the head eneity is: {head_entity}, and the tail entity is: {tail_entity}.\n \
            the extract response is: {extract_response},\n \
            the refine response is: {refine_response},\n \
            the confidence response is: {confidence_response},\n \
            the relation extraction response is: {relation_extraction_response}\n"
            )
        return extract_response, refine_response, confidence_response, relation_extraction_response
    
    def extract_gpt_joint_chain(self, sentence: str, head_entity: str, tail_entity: str):
        extract_refine_prompt = self.task_config.joint.extract_refine_prompt.format(
            sentence=sentence, head_entity=head_entity, tail_entity=tail_entity, relation_labels=self.labels
        )
        extract_refine_prompt = extract_refine_prompt + self.task_config.joint.extract_refine_command
        extract_refine_response = self.get_gpt_asnwer(
            extract_refine_prompt, self.task_config.joint.max_tokens_extract_refine
        )
        
        confidence_prompt = self.task_config.joint.confidence_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            entity_types=extract_refine_response,
        )
        confidence_prompt = confidence_prompt + self.task_config.joint.confidence_command
        confidence_response = self.get_gpt_asnwer(
            confidence_prompt, self.task_config.joint.max_tokens_confidence
        )
        
        relation_extraction_prompt = self.task_config.joint.relation_extraction_prompt.format(
            input_text=sentence,
            head_entity=head_entity,
            tail_entity=tail_entity,
            refined_relation_labels=extract_refine_response,
            relationship_confidence_scores=confidence_response,
        )
        relation_extraction_prompt = relation_extraction_prompt + self.task_config.joint.relation_extraction_command
        relation_extraction_response = self.get_gpt_asnwer(
            relation_extraction_prompt,
            self.task_config.joint.max_tokens_relation_extraction,
        )
        
        logging.info(
            f"the sentence is: {sentence}, and the head eneity is: {head_entity}, and the tail entity is: {tail_entity}.\n \
            the extract refine response is: {extract_refine_response},\n \
            the confidence response is: {confidence_response},\n \
            the relation extraction response is: {relation_extraction_response}\n"
            )
        return extract_refine_response, confidence_response, relation_extraction_response
            
    
    def print_results(self, result: Dict[str, str]):
        for k, v in result.items():
            print(f"{k}: {v}")
            print("--------------------\n")
        print("=====================================\n")
        
    def run_chain(self):
        all_results = []
        result_file_path = f"results/{self.model_id}_{self.task}_{self.setting}_test.json"
        print(self.labels)
        
        for i, rec in enumerate(self.data):
            if self.task_config.id == "wiki":
                sentence, head_entity, tail_entity, relation = self.process_wiki_record(rec)
            else:
                sentence, head_entity, tail_entity, relation = self.process_record(rec)

            if self.model_config.is_protected:
                if self.setting == "sep":
                    extract_response, refine_response, confidence_response, relation_extraction_response = self.extract_llama_sep_chain(
                        sentence, head_entity, tail_entity
                    )
                elif self.setting == "joint":
                    extract_refine_response, confidence_response, relation_extraction_response = self.extract_llama_joint_chain(
                        sentence, head_entity, tail_entity
                    )
                    result = {
                        "sentence": sentence,
                        "head_entity": head_entity,
                        "tail_entity": tail_entity,
                        "relation": relation,
                        "extract_refine_response": extract_refine_response,
                        "confidence_response": confidence_response,
                        "relation_extraction_response": relation_extraction_response,
                    }
            elif self.model_config.is_gpt:
                if self.setting == "sep":
                    extract_response, refine_response, confidence_response, relation_extraction_response = self.extract_gpt_refinement_chain(
                        sentence, head_entity, tail_entity
                    )
                elif self.setting == "joint":
                    extract_refine_response, confidence_response, relation_extraction_response = self.extract_gpt_joint_chain(
                        sentence, head_entity, tail_entity
                    )
                    result = {
                        "sentence": sentence,
                        "head_entity": head_entity,
                        "tail_entity": tail_entity,
                        "relation": relation,
                        "extract_refine_response": extract_refine_response,
                        "confidence_response": confidence_response,
                        "relation_extraction_response": relation_extraction_response,
                    }
            elif not self.model_config.is_protected and not self.model_config.is_gpt:  # MIXTRAL
                if self.setting == "sep":
                    extract_response, refine_response, confidence_response, relation_extraction_response = self.extract_mixtral_sep_chain(
                        sentence, head_entity, tail_entity
                    )
                else:
                    print("Not implemented yet")
                    continue
            
            if self.setting == "sep":
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

            all_results.append(result)
            
            if (i + 1) % 10 == 0:
                with open(result_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(all_results, json_file, indent=2, ensure_ascii=False)
            
            self.print_results(result)
        
        with open(result_file_path, "w", encoding="utf-8") as json_file:
            json.dump(all_results, json_file, indent=2, ensure_ascii=False)