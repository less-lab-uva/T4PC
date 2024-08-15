import time
import subprocess

def check_slurm_queue(job_name):
    while True:
        # Run 'squeue' command and filter by job name
        result = subprocess.run(['squeue', '-n', job_name], stdout=subprocess.PIPE)

        # Get the output and decode it from bytes to string
        output = result.stdout.decode('utf-8')

        # Split the output into lines and count them, subtract 1 for the header line
        queue_size = len(output.strip().split('\n')) - 1

        if queue_size > 500:
            time.sleep(60)
            print(f"Queue size: {queue_size}. Sleeping for 60 seconds...")
        else:
            break

def main():
    job_name = "rq1"
    check_slurm_queue(job_name)

if __name__ == "__main__":
    main()