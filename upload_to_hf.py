import sys
import re
from pathlib import Path
# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("HF_TOKEN environment variable not found.")
    TOKEN = input("Please enter your Hugging Face WRITE token: ").strip()
    if not TOKEN:
        print("Error: Hugging Face token is required.")
        sys.exit(1)

def main():
    api = HfApi(token=TOKEN)
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
        
    repo_id = f"{username}/talentlens-data"
    
    # Create the dataset repo (safe if it already exists)
    print(f"Creating/verifying dataset repository: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="dataset", token=TOKEN, exist_ok=True, private=False)
    
    # Upload candidates.jsonl
    print("Uploading data/candidates.jsonl (~487MB)...")
    api.upload_file(
        path_or_fileobj="data/candidates.jsonl",
        path_in_repo="candidates.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    # Upload embeddings_cache.npz
    print("Uploading data/embeddings_cache.npz (~160MB)...")
    api.upload_file(
        path_or_fileobj="data/embeddings_cache.npz",
        path_in_repo="embeddings_cache.npz",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
    print("\nFile upload completed successfully!")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    
    # Now update app.py with this repo ID
    app_py_path = Path("app.py")
    if app_py_path.exists():
        content = app_py_path.read_text(encoding="utf-8")
        new_content, count = re.subn(
            r'HF_REPO_ID\s*=\s*["\'][^"\']+["\']',
            f'HF_REPO_ID = "{repo_id}"',
            content
        )
        if count > 0:
            app_py_path.write_text(new_content, encoding="utf-8")
            print(f"Updated HF_REPO_ID in app.py to: {repo_id}")
        else:
            print("Warning: HF_REPO_ID variable not found in app.py")

if __name__ == "__main__":
    main()
