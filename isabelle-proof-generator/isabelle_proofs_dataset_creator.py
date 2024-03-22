'''
It takes the PISA dataset https://aitp-conference.org/2021/abstract/paper_17.pdf as input and converts it to a
new dataset with pairs: (lemma+additional data, proof)
'''

import os
from utils import read_json_file, write_json_file

irrelevant_steps = ['text', 'subsection']
end_proofs = ['done', 'qed']

def is_irrelevant_step(step_to_check: str):
    return len([step for step in irrelevant_steps if step_to_check.startswith(step)]) > 0

def is_end_proof(step_to_check: str):
    return len([step for step in end_proofs if step_to_check.startswith(step)]) > 0

def is_not_proof(step_to_check: str):
    return not step_to_check.startswith('proof')

def get_dataset_items_from_extraction_file(extraction_file_path: str):
    extraction_file_content = read_json_file(extraction_file_path)

    translations = extraction_file_content.get('translations')

    proofs = []
    current_proof = ''
    current_lemma = ''
    proof_started = False

    for translation in translations:
        step = translation[1]

        if is_irrelevant_step(step):
            continue

        if not proof_started and is_not_proof(step):
            current_lemma = f'{current_lemma}{step} '
            continue

        proof_started = True
        current_proof = f'{current_proof}{step} '

        if is_end_proof(step):
            proofs.append({'lemma': current_lemma, 'proof': current_proof})
            current_proof = ''
            current_lemma = ''
            proof_started = False
    
    return proofs

def get_dataset_items_from_extraction_folder(extraction_folder_path: str):
    proofs = []
    for file_name in os.listdir(extraction_folder_path):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(extraction_folder_path, file_name)
        if os.path.isfile(file_path):
            proofs.extend(get_dataset_items_from_extraction_file(file_path))

    return proofs

problems_folder = os.getenv('PROBLEMS_FOLDER')
output_file = os.getenv('OUTPUT_FILE_DATASET')

proofs = [get_dataset_items_from_extraction_folder(os.path.join(problems_folder, str(folder))) 
          for folder in os.listdir(problems_folder) 
                   if os.path.isdir(os.path.join(problems_folder, str(folder)))]
flattened_proofs = [element for sublist in proofs for element in sublist]

write_json_file(output_file, { 'proofs': flattened_proofs })