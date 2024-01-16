# from data.data_processor import read_json_txt_file
from src.core_chain import ChainofRefinement
from src.labels import TACRED_LABELS, TACREV_LABELS, RETACRED_NL_LABELS
from data.data_processor import read_json_txt_tacrev, read_json_txt_file, filter_and_sample_records
import json

path = "./dataset/retacred/test.txt"
access_token = "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"

experiment = "TACRED"

file_path_mapping = {
    "TACRED": "./dataset/tacred/test.txt",
    "TACREV": "./dataset/tacrev/test.txt",
    "RETACRED": "./dataset/retacred/test.txt",
}

if experiment == "TACRED":
    data = read_json_txt_file(file_path_mapping['TACRED'])
else:
    data = read_json_txt_tacrev(file_path_mapping['RETACRED'])
# test_data = []

# for rec in data[:20]:
#     print(rec['relation'])
#     if rec['relation'] != 'no_relation':
#         test_data.append(rec)
#     else:
#         continue

data = filter_and_sample_records(data, labels=TACRED_LABELS, sample_size=1000)
# data = data[:30]

chain = ChainofRefinement(
    # model_id = "llama2_70b",
    model_id="gpt",
    top_p=0.9,
    temperature=0.001,
    task='TACRED',
    setting='sep',
    data=data,
    access_token=access_token,
    # labels = TACRED_LABELS
)

chain.run_chain()
