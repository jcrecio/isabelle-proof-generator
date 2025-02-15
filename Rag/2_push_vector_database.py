from huggingface_hub import HfApi
from huggingface_hub import login

login()
repo_id = "jcrecio/afp_rag"
api = HfApi()

api.upload_folder(
    folder_path="isabelle-hol-vectors",
    repo_id=repo_id,
    repo_type="model",
)
