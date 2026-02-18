import os
import sys

def check_file(filepath):
    if os.path.exists(filepath):
        print(f"[OK] Found {filepath}")
        return True
    else:
        print(f"[FAIL] Missing {filepath}")
        return False

def check_dir(dirpath):
    if os.path.isdir(dirpath):
        print(f"[OK] Found directory {dirpath}")
        return True
    else:
        print(f"[FAIL] Missing directory {dirpath}")
        return False

def main():
    print("Verifying Project Setup...")
    
    # Check key files
    files_to_check = [
        "README.md",
        "requirements.txt",
        "src/utils.py",
        "src/0_download_data.py",
        "src/1_preprocess.py",
        "src/2_train.py",
        "src/3_convert.py",
        "src/4_generate_synthetic_data.py"
    ]
    
    # Check directories
    dirs_to_check = [
        "src",
        "models",
        "data"
    ]

    all_passed = True
    
    for f in files_to_check:
        if not check_file(f):
            all_passed = False
            
    for d in dirs_to_check:
        if not check_dir(d):
            # data and models might not exist initially, just warn? 
            # But ensure_dir in scripts should create them. 
            # Let's count them as failed if missing for a complete setup check.
            all_passed = False

    if all_passed:
        print("\nSUCCESS: Project structure looks correct.")
    else:
        print("\nWARNING: Some files or directories are missing.")

if __name__ == "__main__":
    main()
