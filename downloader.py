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
    parser.add_argument("--option", type=str, choices=["controlled_experiment_results", "case_study_results", "dataset", "tcp_model"], default=None)
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
    elif args.option == "dataset":
        urls = [
            "https://zenodo.org/records/13327114/files/town01.7z?download=1",
            "https://zenodo.org/records/13327297/files/town01_addition.7z?download=1",
            "https://zenodo.org/records/13327297/files/town01_val.7z?download=1",
            "https://zenodo.org/records/13327114/files/town02.7z?download=1",
            "https://zenodo.org/records/13327297/files/town02_val.7z?download=1",
            "https://zenodo.org/records/13327297/files/town04.7z?download=1",
            "https://zenodo.org/records/13323713/files/town04_addition.7z?download=1",
            "https://zenodo.org/records/13323713/files/town04_val.7z?download=1",
            "https://zenodo.org/records/13323713/files/town05.7z?download=1",
            "https://zenodo.org/records/13323713/files/town05_addition.7z?download=1",
            "https://zenodo.org/records/13327114/files/town05_val.7z?download=1",
            "https://zenodo.org/records/13323713/files/town07.7z?download=1",
            "https://zenodo.org/records/13327297/files/town07_val.7z?download=1",
            "https://zenodo.org/records/13327297/files/town10.7z?download=1",
            "https://zenodo.org/records/13327114/files/town10_addition.7z?download=1",
            "https://zenodo.org/records/13323713/files/town10_val.7z?download=1"
        ]
        output_files = [
            "town01.7z",
            "town01_addition.7z",
            "town01_val.7z",
            "town02.7z",
            "town02_val.7z",
            "town04.7z",
            "town04_addition.7z",
            "town04_val.7z",
            "town05.7z",
            "town05_addition.7z",
            "town05_val.7z",
            "town07.7z",
            "town07_val.7z",
            "town10.7z",
            "town10_addition.7z",
            "town10_val.7z"
        ]
        for url, output_file in zip(urls, output_files):
            download_file(url, output_file)
            if not os.path.exists("./carla_dataset/"):
                os.makedirs("./carla_dataset/")
            extract_file(output_file, "./carla_dataset/")
            remove_file(output_file)
            break
    elif args.option == "tcp_model":
        url = "https://zenodo.org/records/13327702/files/best_model.ckpt?download=1"
        output_file = "./case_study/TCP/best_model.ckpt"
        download_file(url, output_file)
    else:
        raise ValueError("Invalid option")

if __name__ == "__main__":
    main()