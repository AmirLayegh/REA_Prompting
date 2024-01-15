import json
import ast

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
# def process_record(record):
#     sentence = record['token']
#     sentence = " ".join(sentence)
#     head_entity = record['h']['name']
#     tail_entity = record['t']['name']
#     relation = record['relation']
#     return sentence, head_entity, tail_entity, relation

