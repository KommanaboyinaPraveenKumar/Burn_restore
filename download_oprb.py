import os
import tarfile
from huggingface_hub import hf_hub_download

def download_and_extract(repo_id, filename, extract_dir):
    print(f"Downloading {filename} from {repo_id}...")
    tar_path = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=filename)
    
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting {tar_path} into {extract_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Avoid zip slip by checking paths if needed, or simply extract all for trusted datasets
        tar.extractall(path=extract_dir)
    print(f"Extraction complete for {filename}.")

if __name__ == "__main__":
    repo_id = "kpurkayastha/OPRB"
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "oprb_train")
    
    img_dir = os.path.join(base_dir, "img")
    gt_dir = os.path.join(base_dir, "gt")
    
    download_and_extract(repo_id, "train/text/img.tar.gz", img_dir)
    download_and_extract(repo_id, "train/text/gt.tar.gz", gt_dir)
    
    print(f"Successfully downloaded and extracted OPRB dataset to {base_dir}")
