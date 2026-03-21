from pathlib import Path

from huggingface_hub import HfApi


def push_directory_to_hub(local_dir: str, repo_id: str, commit_message: str = "Upload model") -> str:
    api = HfApi()
    folder = Path(local_dir)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=commit_message,
        ignore_patterns=["checkpoints", "checkpoints/**", "_autotune", "_autotune/**"],
        delete_patterns=["checkpoints", "checkpoints/**", "_autotune", "_autotune/**"],
    )
    return f"https://huggingface.co/{repo_id}"
