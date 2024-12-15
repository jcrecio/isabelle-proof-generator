'''
It takes the PISA dataset https://aitp-conference.org/2021/abstract/paper_17.pdf as input and converts it to a
new JSON dataset with triplets: (context, theorem statement, proof)
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
        theorem_statement = f'''{theorem_statement}
{lemma_step}'''

    for index in range(last_lemma_index + 1, len(translations)):
        proof_step = translations[index][1]
        proof = f'''{proof}
{proof_step} '''
    
    return [theorem_statement, proof]

def read_abstract_from_tex_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        start_tag = '\\begin{abstract}'
        end_tag = '\\end{abstract}'
        start_index = content.find(start_tag)
        end_index = content.find(end_tag)
        if start_index != -1 and end_index != -1:
            start_index += len(start_tag)
            return content[start_index:end_index].strip()
        
    return 'NO_CONTEXT'

def get_problem_context(afp_folder):
    abstract_tex_path = os.path.join(afp_folder, 'document', 'abstract.tex')
    if os.path.exists(abstract_tex_path):
        return read_abstract_from_tex_file(abstract_tex_path)
    
    root_tex_path = os.path.join(afp_folder, 'document', 'root.tex')
    if os.path.exists(root_tex_path):
        return read_abstract_from_tex_file(root_tex_path)

    return 'NO_CONTEXT'

def get_dataset_items(afp_folder: str, extraction_folder_path: str, with_context: bool):
    proofs = []
    for file_name in os.listdir(extraction_folder_path):
        if not file_name.endswith(".json"):
            continue

        context = 'NO_CONTEXT'
        if with_context == 'True':
            context = get_problem_context(afp_folder)

        file_path = os.path.join(extraction_folder_path, file_name)
        if os.path.isfile(file_path):
            [theorem_statement, proof] = get_lemma_proof_for_file(file_path)
            if theorem_statement != '' and proof != '':
                proofs.append({'context': context, 'theorem_statement': theorem_statement, 'proof': proof})

    return proofs

problems_folder = os.getenv('PROBLEMS_FOLDER')
afp_folder = os.getenv('AFP_FOLDER')
output_file = os.getenv('OUTPUT_FILE_DATASET')
with_context = os.getenv('WITH_CONTEXT')

proofs = []
for item in os.listdir(problems_folder):
    proofs = proofs + get_dataset_items(os.path.join(afp_folder, str(item)),os.path.join(problems_folder, str(item)), with_context)

write_json_file(output_file, { 'proofs': proofs })