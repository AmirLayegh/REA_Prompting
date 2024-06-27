
# REA Prompt for Zero-Shot Relation Extraction

This repository contains the implementation and experimental framework for the paper “REA: Refine-Estimate-Answer Prompting for Zero-Shot Relation Extraction,” focusing on enhancing machine understanding of unstructured text without prior training data.

![REA Zero-Shot Relation Extraction Process](/rea.jpg)


## Quick Start 🚀

### Prerequisites 📋

- Python 3.10 or higher
- CUDA-capable GPU for efficient model computation

### Installation 🔧

1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a Python virtual environment**:
   ```
   conda create -n REA python=3.10
   conda activate REA
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Running the Experiments 🧪

### Local Execution

Execute the experiments locally using:
```
python main.py --model_id your_model_id --top_p your_top_p --temperature your_temperature --task your_task --setting your_setting --access_token your_access_token --rel_size your_rel_size
```

### 📦 Cluster Execution with SLURM

For cluster environments using SLURM:
1. **Prepare the Apptainer container** (`SIF_FILE.sif`) in your directory.
2. **Build the Apptainer container** using the provided `my_apptainer.def` file:
   ```bash
   # Build the Apptainer container
   apptainer build my_container.sif my_apptainer.def
   ```
3. **Run the REA experiments within the Apptainer container** by submitting the SLURM `chain_job`:
   ```bash
   sbatch chain_job.sh
   ```

This script leverages a single GPU and runs experiments for up to 48 hours.

## Data 📊

Utilize these datasets:
- TACRED
- FewRel
- Wiki

Place your dataset files in designated directories as outlined in `main.py`.

## Contributing 🤝

Contributions are welcome! Please submit pull requests or open issues for improvements or bug fixes.

## Citation 📄

This paper has been presented but is not yet published. You can access the full text on the author's personal page [here](https://amirlayegh.github.io/files/REA_Prompting.pdf).
