# T4PC

## TCP property conformance improvement for stop signs
This video shows **TCP fine-tuned without property loss** running in one of the evaluation routes in CARLA, and we can see how it **runs a stop sign.**.

![alt text](video_not_stopping.gif)

This video shows **TCP fine-tuned with T4PC** running in the same evaluation route in CARLA, but in this case we can see that TCP safely **stops at the stop sign**.

![alt text](video_stopping.gif)


## Reproduce results
### Conda environment
1. Create a conda environment
```bash
conda env create -f environment.yml --prefix .t4pc
```

2. Activate the environment
```bash
conda activate .t4pc
```

### RQ1, RQ2, and RQ3
1. Download the violations csv files by running:
```bash
python downloader.py --option controlled_experiment_results
```

2. Run the following script to plot the results:
```bash
python controlled_experiment/plot.py
```
Figure 3 in the paper will be generated at `./controlled_experiment/results.png`.

### TCP Controlled Experiment
1. Download the results.json files by running:
```bash
python downloader.py --option case_study_results
```

2. To render the full table with the results, execute the cells in the Jupyter notebook `case_study/results_summary.ipynb`.



## Run full experiments: RQ1, RQ2, and RQ3

### Conda environment
1. Create a conda environment
```bash
conda env create -f environment.yml --prefix .t4pc
```

2. Activate the environment
```bash
conda activate .t4pc
```

### Download the dataset
Download the dataset by running:
```bash
python downloader.py --option dataset
```

### Defining directories paths
You need to define the variables in the .env sample file. The variables are: ROOT_DIR, DATASET_DIR, CARLA_DIR.

### Execution scripts
The scripts are designed to run in a SLURM server. The scripts need to be executed sequentially, consequently, execute each of the commands after all the previous SLURM jobs are completed. To launch the experiments, execute the following command:
```bash
python controlled_experiment/slurm_scripts/rq1_runner.py
python controlled_experiment/slurm_scripts/rq2_runner.py
python controlled_experiment/slurm_scripts/rq3_runner.py
```
Note that some of the SBATCH parameters inside `controlled_experiment/rq1/sbatch_rq1.sh`, `controlled_experiment/rq2/sbatch_rq2.sh`, and `controlled_experiment/rq3/sbatch_rq3.sh` may need to be adjusted to the specific server configuration.

### Plot scripts
To plot the results. Run the following script:
```bash
python controlled_experiment/plot.py
```


## Run full case study: TCP

### Conda environment
1. Create a conda environment
```bash
conda env create -f environment.yml --prefix .t4pc
```

2. Activate the environment
```bash
conda activate .t4pc
```

### Download Carla
Download CARLA 0.9.10.1 and the additional maps by running:
```bash
mkdir carla
cd carla
wget https://tiny.carla.org/carla-0-9-10-1-linux
wget https://tiny.carla.org/additional-maps-0-9-10-1-linux
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
```

### Download files
1. Download the dataset by running:
```bash
python downloader.py --option dataset
```
2. Download the TCP base model by running:
```bash
python downloader.py --option tcp_model
```

### Defining directories paths
You need to define the variables in the .env sample file. The variables are: ROOT_DIR, DATASET_DIR, CARLA_DIR.

### Training and benchmark scripts
The scripts are designed to run in a SLURM server. The file `case_study/slurm_scripts/experiment.sh` contains the commands to run the training and benchmarking for all models. To launch the experiments, execute the following command:
```bash
python case_study/slurm_scripts/experiment.sh
```
Note that some of the SBATCH parameters inside `case_study/slurm_scripts/sbatch_training.sh` and `case_study/slurm_scripts/sbatch_benchmark.sh` may need to be adjusted to the specific server configuration.
