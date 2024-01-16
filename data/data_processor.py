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

def read_json_txt_tacrev(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        records = [ast.literal_eval(line) for line in file]

    return records

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

# def process_record(record):
#     sentence = record['token']
#     sentence = " ".join(sentence)
#     head_entity = record['h']['name']
#     tail_entity = record['t']['name']
#     relation = record['relation']
#     return sentence, head_entity, tail_entity, relation

