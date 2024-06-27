import argparse
from src.rea_chain import ReaChain, ChainConfig
from src.utils import MODEL_MAPPING, TASK_MAPPING
from src.labels import TACRED_LABELS
from data.data_processor import (
    read_json_txt_file,
    filter_and_sample_records,
    sample_fewrel_dataset,
    read_wiki_file,
    filter_and_sample_records_wiki,
)

m=10
FILE_PATH_MAPPING = {
    "TACRED": "./data/TACRED/test.json",
    "FewRel": "./data/FewRel/test.json",
    "wiki": "./data/wiki/test_m={}.json".format(m),
}

def parse_arguments():
    parser = argparse.ArgumentParser(description= "REA Prompt for Zero-Shot Relation Extraction")
    parser.add_argument("--model_id", type=str, default="gpt", help="Model ID")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top p for sampling from the model")
    parser.add_argument("--temperature", type=float, default=0.001, help="Temperature for sampling from the model")
    parser.add_argument("--task", type=str, default="wiki", help="Task to run the chain on")
    parser.add_argument("--setting", type=str, default="sep", help="Setting to run the chain on")
    parser.add_argument("--access_token", type=str, default="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", help="Access token for the Huggingface API")
    parser.add_argument("--rel_size", type=int, default=15, help="Number of relations to sample from the FewRel and Wiki datasets")
    return parser.parse_args()

def load_data(task, rel_size):
    if task == "TACRED":
        data = read_json_txt_file(FILE_PATH_MAPPING[task])
        data = filter_and_sample_records(data, rel_size)
        labels = TACRED_LABELS
    elif task == "FewRel":
        data = read_json_txt_file(FILE_PATH_MAPPING['FewRel'])
        data = sample_fewrel_dataset(data, rel_size)
        labels = list(set(record['relation'] for record in data))
        data = filter_and_sample_records(data, labels=labels, sample_size=1000)
    elif task == "wiki":
        data = read_wiki_file(FILE_PATH_MAPPING['wiki'])
        labels = list(set(relation['type'] for record in data for relation in record['relations']))
        data = filter_and_sample_records_wiki(data, labels=labels, sample_size=500)
    else:
        raise ValueError("Invalid task")
    return data, labels

def main():
    args = parse_arguments()
    
    data, labels = load_data(args.task, args.rel_size)
    config = ChainConfig(
        model_id=args.model_id,
        top_p=args.top_p,
        temperature=args.temperature,
        task=args.task,
        setting=args.setting,
        access_token=args.access_token,
    )
    
    chain = ReaChain(
        config=config,
        data=data,
        labels=labels,
        access_token=args.access_token,)
    
    chain.run_chain()
    
if __name__ == "__main__":
    main()
    
        
    

