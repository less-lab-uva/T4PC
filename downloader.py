import argparse
import subprocess
import shutil
import os

# Function to download a file using wget
def download_file(url, output_file):
    try:
        subprocess.run(["wget", url, "-O", output_file], check=True)
        print(f"Downloaded {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")

# Function to extract a file using 7z
def extract_file(archive_file, output_dir="."):
    try:
        subprocess.run(["7z", "x", archive_file, f"-o{output_dir}"], check=True)
        print(f"Extracted {archive_file} to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting file: {e}")

# Function to remove a file
def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"Removed file: {file_path}")
    except OSError as e:
        print(f"Error removing file: {e}")

def main():
    # Define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, choices=["controlled_experiment_results", "case_study_results"], default=None)
    args = parser.parse_args()
    
    if args.option == "controlled_experiment_results":
        urls = ["https://zenodo.org/records/13312460/files/rq1_violations.7z?download=1",
                "https://zenodo.org/records/13312460/files/rq2_violations.7z?download=1",
                "https://zenodo.org/records/13312460/files/rq3_violations.7z?download=1"]
        output_files = ["rq1_violations.7z", "rq2_violations.7z", "rq3_violations.7z"]
        for url, output_file in zip(urls, output_files):
            download_file(url, output_file)
            extract_file(output_file)
            remove_file(output_file)

    elif args.option == "case_study_results":
        url = "https://zenodo.org/records/13312460/files/results_summary.7z?download=1"
        output_file = "case_study_results.7z"
        download_file(url, output_file)
        extract_file(output_file, "./case_study/")
        remove_file(output_file)
    else:
        raise ValueError("Invalid option")

if __name__ == "__main__":
    main()