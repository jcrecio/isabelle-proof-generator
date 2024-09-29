'''
It takes the PISA dataset https://aitp-conference.org/2021/abstract/paper_17.pdf as input and converts it to a
new JSON dataset with pairs: (theorem statement, proof)
'''

import os
from utils import read_json_file, write_json_file

def find_lemma_index_in_translations(lemma, translations):
    for index, pair in enumerate(translations):
        if pair[1] == lemma:
            return index
    return -1

def get_lemma_proof_for_file(extraction_file_path: str):
    file_content = read_json_file(extraction_file_path)
    translations = file_content.get('translations')
    if translations is None or len(translations) == 0:
        return '',''
    
    lemma = file_content.get('problem_names')[-1]

    last_lemma_index = find_lemma_index_in_translations(lemma, translations)

    theorem_statement = ''
    proof = ''

    for index in range(0, last_lemma_index):
        lemma_step = translations[index][1]
        theorem_statement = f'{theorem_statement}\\n{lemma_step} '

    for index in range(last_lemma_index + 1, len(translations)):
        proof_step = translations[index][1]
        proof = f'{proof}\\n{proof_step} '
    
    return [theorem_statement, proof]

def get_dataset_items_from_extraction_folder(extraction_folder_path: str):
    proofs = []
    for file_name in os.listdir(extraction_folder_path):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(extraction_folder_path, file_name)
        if os.path.isfile(file_path):
            [theorem_statement, proof] = get_lemma_proof_for_file(file_path)
            if theorem_statement != '' and proof != '':
                proofs.append({'theorem_statement': theorem_statement, 'proof': proof})

    return proofs

problems_folder = os.getenv('PROBLEMS_FOLDER')
output_file = os.getenv('OUTPUT_FILE_DATASET')

proofs = [get_dataset_items_from_extraction_folder(os.path.join(problems_folder, str(folder))) 
          for folder in os.listdir(problems_folder) 
                   if os.path.isdir(os.path.join(problems_folder, str(folder)))]
flattened_proofs = [element for sublist in proofs for element in sublist]

write_json_file(output_file, { 'proofs': flattened_proofs })