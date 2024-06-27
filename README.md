
```markdown
# REA Prompt for Zero-Shot Relation Extraction

This repository contains code for running experiments on zero-shot relation extraction for the paper “REA: Refine-Estimate-Answer Prompting for Zero-Shot Relation Extraction”.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (for GPU acceleration)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment:
   ```
   conda create -n REA python=3.10
   conda activate REA
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Experiments

### Local Execution

To run the experiments locally, use the following command:

```
python main.py --model_id <model_id> --top_p <top_p> --temperature <temperature> --task <task> --setting <setting> --access_token <access_token> --rel_size <rel_size>
```

Arguments:
- `--model_id`: Model ID (default: "gpt")
- `--top_p`: Top p for sampling from the model (default: 0.5)
- `--temperature`: Temperature for sampling from the model (default: 0.001)
- `--task`: Task to run the chain on (default: "wiki")
- `--setting`: Setting to run the chain on (default: "sep")
- `--access_token`: Access token for the Huggingface API
- `--rel_size`: Number of relations to sample from the FewRel and Wiki datasets (default: 15)

### Cluster Execution with SLURM

If you're running on a cluster with SLURM, you can use the provided `chain_job.sh` script:

1. Ensure you have the necessary Apptainer container (SIF_FILE.sif) in your working directory.

2. Submit the job to SLURM:
   ```
   sbatch chain_job.sh
   ```

This script will run the experiment on a single GPU for up to 48 hours.

## Data

The code supports three datasets:
- TACRED
- FewRel
- Wiki

Ensure that your data files are placed in the correct directories as specified in the `FILE_PATH_MAPPING` in `main.py`.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes.

## Citation