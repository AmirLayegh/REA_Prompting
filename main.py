#from data.data_processor import read_json_txt_file
from src.core_chain import ChainofRefinement
from data.data_processor import read_json_txt_tacrev, read_json_txt_file
import json

path = "./dataset/retacred/test.txt"
access_token = "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"

file_path_mapping = {
    "TACRED": "./dataset/tacred/test.txt",
    "TACREV": "./dataset/tacrev/test.txt",
    "RETACRED": "./dataset/retacred/test.txt",
}


#data = read_json_txt_file(file_path_mapping['TACREV'])
data = read_json_txt_tacrev(file_path_mapping['RETACRED'])
test_data = []

for rec in data[:20]:
    print(rec['relation'])
    if rec['relation'] != 'no_relation':
        test_data.append(rec)
    else:
        continue

#data = data[:1000]

chain = ChainofRefinement(
    #model_id = "llama2_70b",
    model_id= "gpt",
    top_p = 0.9,
    temperature = 0.07,
    task = 'ReTACRED',
    setting = 'sep',
    data = test_data,
    access_token = access_token,
    #labels = TACRED_LABELS
    )

chain.run_chain()
    
    
    