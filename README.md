# Count Bridges: A Framework for Discrete Distribution Modeling

This repository contains the official implementation for the paper on **Count Bridges**, a novel method for modeling discrete distributions. The codebase provides a flexible framework for experimenting with various generative models, including Conditional Flow Matching (CFM), Discrete Flow Matching (DFM), and our proposed Count Bridge models.

## Key Features

- **Flexible Modeling**: Implements several generative models:
    - **Count Bridges**: Our novel approach for discrete data.
    - **Conditional Flow Matching (CFM)**: A powerful baseline for continuous data.
    - **Discrete Flow Matching (DFM)**: An adaptation of flow matching for discrete data.
- **Diverse Datasets**: Includes support for both synthetic and real-world datasets:
    - **Synthetic Datasets**:
        - 8 Gaussians to 2 Moons (on integers)
        - Low-Rank Gaussian Mixtures
    - **Real-world Applications**:
        - Nucleotide-level single-cell modeling
        - Spatial deconvolution of simulated Visium spots
- **Modular Architecture**: Built with a modular design, allowing for easy extension and experimentation with new models, architectures, and datasets.
- **Reproducibility**: Leverages [Hydra](https://hydra.cc/) for configuration management, ensuring reproducible experiments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd counting_flows
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n counting_flows python=3.10
    conda activate counting_flows
    ```

3.  **Install dependencies:**
    The project uses PyTorch and CuPy for GPU acceleration. Please ensure you have a compatible CUDA version installed.

    ```bash
    pip install -r requirements.txt
    ```
    *Note: You might need to adjust the `cupy-cuda12x` version in `requirements.txt` to match your CUDA toolkit version.*

## Codebase Structure

The project is organized as follows:

```
├── configs/                # Hydra configuration files
│   ├── architecture/       # Network architectures (e.g., MLP, UViT)
│   ├── bridge/             # Bridge sampling methods (CFM, DFM, Skellam)
│   ├── dataset/            # Dataset configurations
│   └── ...                 # Other configs (optimizer, scheduler, etc.)
├── architecture/           # Neural network architecture implementations
├── bridges/                # Bridge sampling implementations (PyTorch, CuPy, NumPy)
├── datasets/               # Data loading and preprocessing scripts
├── models/                 # Model definitions and loss functions
├── results/                # Analysis notebooks and saved results/figures
├── main.py                 # Main script to run experiments with Hydra
├── training.py             # Core training loop
└── evaluate.py             # Evaluation and sampling logic
```

## Running Experiments

Experiments are managed through Hydra. You can run experiments by specifying a configuration file and overriding parameters from the command line.

The main entry point is `main.py`.

### Basic Training Example

To train a model with the default configuration (`configs/config.yaml`), simply run:

```bash
python main.py
```

This will train a model on the `LowRankGaussianMixtureDataset` using the `skellam` bridge and an `mlp` architecture.

### Overriding Configuration

Hydra allows you to easily override any configuration parameter from the command line.

**Example: Training a CFM model on the Discrete Moons dataset:**

```bash
python main.py bridge=cfm dataset=discrete_moons
```

**Example: Changing the neural network architecture:**

```bash
python main.py architecture=uvit_small
```

**Example: Running a deconvolution experiment:**

To run a deconvolution experiment, you can use the `config_deconv.yaml` config file.

```bash
python main.py -cn config_deconv
```

### Multi-run Experiments

Hydra's multi-run feature is useful for hyperparameter sweeps.

**Example: Sweeping over different learning rates:**

```bash
python main.py --multirun optimizer.learning_rate=1e-3,1e-4,1e-5
```

Outputs from experiments (checkpoints, logs, and plots) are saved in the `outputs/` or `multirun/` directory, organized by date and time.

## Configuration System

The configuration is structured hierarchically using Hydra. The main config file is `configs/config.yaml`, which sets the defaults.

Key configuration groups:

-   `bridge`: Defines the type of bridge to use (e.g., `cfm`, `dfm`, `skellam`).
-   `model`: Specifies the model and loss function (e.g., `energy_score`, `mse`).
-   `architecture`: Chooses the neural network architecture (e.g., `mlp`, `uvit_multimodal`).
-   `dataset`: Selects the dataset for the experiment (e.g., `low_rank_gaussian_mixture`, `discrete_moons`).
-   `training`: Training parameters like number of epochs, batch size, etc.

You can create new configurations by adding YAML files to the respective directories under `configs/`.


