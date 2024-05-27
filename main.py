from src.core_chain import ReaChain
from src.labels import TACRED_LABELS, TACREV_LABELS, RETACRED_NL_LABELS
from data.data_processor import (
    read_json_txt_tacrev, 
    read_json_txt_file, 
    filter_and_sample_records, 
    sample_fewrel_dataset,
    read_wiki_file,
    filter_and_sample_records_wiki,
)
import json
import argparse

file_path_mapping = {
    "TACRED": "./dataset/tacred/test.txt",
    "TACREV": "./dataset/tacrev/test.txt",
    "RETACRED": "./dataset/retacred/test.txt",
    "FewRel": "./dataset/fewrel/fewrel_train.txt",
    "wiki": "./dataset/wiki-zsl/test_m=10.json",
}

def load_and_process_data(task, sample_size, rel_size):
    if task == "TACRED":
        data = read_json_txt_file(file_path_mapping['TACRED'])
        data = filter_and_sample_records(data, labels=TACRED_LABELS, sample_size=sample_size)
    elif task == "FewRel":
        data = read_json_txt_file(file_path_mapping['FewRel'])
        data = sample_fewrel_dataset(data, rel_size)
        labels = list(set([record['relation'] for record in data]))
        data = filter_and_sample_records(data, labels=labels, sample_size=sample_size)
    elif task == "wiki":
        data = read_wiki_file(file_path_mapping['wiki'])
        labels = list(set([relation['type'] for record in data for relation in record['relations']]))
        data = filter_and_sample_records_wiki(data, labels=labels, sample_size=sample_size)
    else:
        raise ValueError(f"Unknown task: {task}")
    return data, labels

def main():
    parser = argparse.ArgumentParser(description="REA Prompting for Zero-Shot Relation Extraction")
    parser.add_argument("-m", "--model_id", type=str, default="gpt", help="Model ID", options=["gpt", "llama2_13b", "llama2_70b", "mistral", "zephyr", "mixtral"])
    parser.add_argument("-p", "--top_p", type=float, default=0.5, help="Top p for sampling from the model")
    parser.add_argument("-temp", "--temperature", type=float, default=0.001, help="Temperature for sampling from the model")
    parser.add_argument("-t", "--task", type=str, default="wiki", help="Task to run the chain on", options=["TACRED", "FewRel", "wiki"])
    parser.add_argument("-s", "--setting", type=str, default="sep", help="Setting to run the chain on", options=["sep", "joint"])
    parser.add_argument("-a", "--access_token", type=str, default="", help="Access token for the Huggingface API")
    parser.add_argument("-m", "--rel_size", type=int, default=15, help="Number of relations to sample from the FewRel and Wiki datasets")
    parser.add_argument("-n", "--sample_size", type=int, default=1000, help="Number of samples to use from the dataset")
    
    args = parser.parse_args()
    
    data, labels = load_and_process_data(args.task, args.sample_size, args.rel_size)
    
    chain = ReaChain(
        model_id=args.model_id,
        top_p=args.top_p,
        temperature=args.temperature,
        task=args.task,
        setting=args.setting,
        data=data,
        access_token=args.access_token,
        labels=labels,
    )
    
    chain.run_chain()

if __name__ == "__main__":
    main()