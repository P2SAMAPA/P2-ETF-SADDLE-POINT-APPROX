import os
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, upload_folder
from config import HF_OUTPUT_REPO, HF_TOKEN, LOCAL_RESULTS_DIR

def upload_results(df_dict: dict, run_id: str = None):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    local_path = os.path.join(LOCAL_RESULTS_DIR, run_id)
    os.makedirs(local_path, exist_ok=True)
    
    for name, df in df_dict.items():
        if df is not None:
            df.to_parquet(os.path.join(local_path, f"{name}.parquet"))
    
    if HF_TOKEN:
        api = HfApi()
        try:
            api.create_repo(repo_id=HF_OUTPUT_REPO, token=HF_TOKEN, exist_ok=True)
        except Exception as e:
            print(f"Repo creation notice: {e}")
        
        # Upload folder without unsupported argument
        upload_folder(
            folder_path=local_path,
            repo_id=HF_OUTPUT_REPO,
            token=HF_TOKEN,
            repo_type="dataset",
            path_in_repo=run_id,
        )
        print(f"Uploaded to {HF_OUTPUT_REPO}/tree/main/{run_id}")
    else:
        print("HF_TOKEN not set, results saved locally only")
    
    return local_path
