#!/usr/bin/env python3

import argparse
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' is not installed.")
    print("Please install it by running: pip install huggingface_hub")
    sys.exit(1)

def download_model(repo_id: str, output_dir: str, download_full: bool):
    """
    Downloads a Hugging Face model. 
    By default excludes unnecessary file formats unless download_full is True.
    """
    
    if download_full:
        ignore_patterns = None
        print(f"Starting FULL download for: '{repo_id}'")
        print(f"Target directory: '{output_dir}'")
        print("Downloading all files (no formats excluded).\n")
    else:
        # Patterns to exclude to save bandwidth and disk space.
        ignore_patterns = [
            "*.h5",          # TensorFlow weights
            "*.msgpack",     # Flax weights
            "*.ot",          # Rust weights
            "*.tflite",      # TensorFlow Lite models
            "*.onnx",        # ONNX weights
            "*coreml*",      # CoreML weights
            "*.ckpt",        # Original non-HF checkpoints
        ]
        print(f"Starting OPTIMIZED download for: '{repo_id}'")
        print(f"Target directory: '{output_dir}'")
        print(f"Excluding files matching: {', '.join(ignore_patterns)}\n")

    try:
        # local_dir_use_symlinks=False ensures the actual files are downloaded 
        # directly into the target folder instead of symlinking to a cache.
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore_patterns,
            resume_download=True
        )
        print(f"\nSuccess! Model downloaded to: {downloaded_path}")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model locally."
    )
    parser.add_argument(
        "repo_id", 
        type=str, 
        help="The Hugging Face model repository ID (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "-d", "--dir", 
        type=str, 
        default=None, 
        help="Output directory. Defaults to ./models/<repo_id>."
    )
    parser.add_argument(
        "-f", "--full", 
        action="store_true", 
        help="Download the full repository. If not set, defaults to excluding TF/Flax/ONNX/CoreML files."
    )
    
    args = parser.parse_args()
    
    # If no directory is provided, default to ./models/<repo_id>
    if args.dir is None:
        # os.path.expanduser correctly resolves '~' to your actual home directory (e.g., /home/user/)
        target_dir = os.path.expanduser(f"./models/{args.repo_id}")
    else:
        target_dir = args.dir

    # Create the directory structure (including parent directories if they don't exist)
    os.makedirs(target_dir, exist_ok=True)
    
    download_model(args.repo_id, target_dir, args.full)
