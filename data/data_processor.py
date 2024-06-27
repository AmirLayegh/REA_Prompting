import json
import ast
from collections import defaultdict
import random

def read_json_txt_file(file_path):
    json_data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    record = json.loads(line)
                    json_data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return json_data


def filter_and_sample_records(json_data, labels, sample_size=1000):
    filtered_records = [record for record in json_data if record['relation'] != 'NA']
    
    label_records = defaultdict(list)

    for record in filtered_records:
        relation = record['relation']
        if relation in labels:
            label_records[relation].append(record)

    sampled_records = []
    
    # Shuffle records for each label
    for label, records in label_records.items():
        random.shuffle(records)
        sampled_records.extend(records[:sample_size // len(labels)])

    return sampled_records

def filter_and_sample_records_wiki(json_data, labels, sample_size=1000):
    label_records = defaultdict(list)
    for record in json_data:
        sentence = record['sentence']
        relations = record['relations']
        for relation in relations:
            relation_type = relation['type']
            if relation_type in labels:
                label_records[relation_type].append(record)
    
    sampled_records = []
    
    for label, records in label_records.items():
        random.shuffle(records)
        sampled_records.extend(records[:sample_size // len(labels)])
    
    return sampled_records
    
    

def save_json_txt_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for record in data:
            json.dump(record, file, ensure_ascii=False)
            file.write('\n')
            
def sample_fewrel_dataset(fewrel_data, m):
    
    unique_relations = set(record['relation'] for record in fewrel_data)
    sampled_relations = random.sample(unique_relations, m)
    sampled_fewrel_data = [record for record in fewrel_data if record['relation'] in sampled_relations]
    #save_json_txt_file(sampled_fewrel_data, output_path)
    return sampled_fewrel_data

def read_wiki_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

