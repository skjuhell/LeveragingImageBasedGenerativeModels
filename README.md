# Leveraging Image-based Generative Adversarial Networks For Time Series Generation

This repository provides the code supplementing _Leveraging Image-based Generative Adversarial
Networks For Time Series Generation_ by Hellermann and Lessmann.

> **Assembled:** July 2025

---

## Contact

For questions or support, please contact **Justin Hellermann** at [justin.hellermann@hu-berlin.com](mailto:justin.hellermann@hu-berlin.de) or Stefan Lessmann at [stefan.lessmann@hu-berlin.de](mailto:stefan.lessmann@hu-berlin.com).

---

## Repository Structure

| Folder/File                       | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `main.py`                         | Main entry point for training, generating, and benchmarking  |
| `data/raw_data/`                  | Folder to place raw input CSVs                               |
| `data/ori/`                       | Stores original sequences                                    |
| `data/syn/`                       | Stores synthetic sequences                                   |
| `results/`                        | Stores benchmarking results, logs, and UMAP visualizations   |
| `requirements_wgan_diffusion.txt` | Dependency file for WGAN and Diffusion models                |
| `requirements_timegan.txt`        | Dependency file for TimeGAN                                  |
| `utils`                           | Utility functions for preprocessing and evaluation           |
| `utils_timegan`                   | Utility functions for preprocessing and evaluation (TimeGAN) |

---

## Computing Environment

- **Operating System:** POSIX-compliant system (Linux recommended)
- **Tested On:**
  - `os.name`: `posix`
  - `platform.system()`: `Linux`
  - `platform.release`: `6.1.x`
- **Python Versions:**
  - WGAN_GP & Diffusion: Python ≥ 3.8
  - TimeGAN: Python ≤ 3.6
- **Expected Runtime:**
  - TimeGAN: ~25 minutes (CPU) per series
  - WGAN_GP: ~10 minutes (GPU recommended) per series
  - Diffusion: ~10 minutes (GPU recommended) per series
- **Hardware Recommendation:**
  - RAM: 16+ GB
  - GPU (optional but recommended for Diffusion): NVIDIA CUDA-enabled GPU

---

## Environment Setup

### For WGAN or Diffusion

```bash
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements_wgan_diffusion.txt
```

### For TimeGAN

```bash
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements_timegan.txt
```

---

## Data Description

- **Format:** CSV with rows as time steps and columns as features
- **Location for Raw Input:**  
  `data/raw_data/{data_name}.csv`
- **Synthetic Output Data:**
  - Generated sequences: `data/syn/`
  - Original sequences: `data/ori/`

### Access & Preprocessing

- **Synthetic datasets** (e.g., `noisy_sine`) are auto-generated and placed in `data/raw_data/` and do not require external access.
- **Public datasets**, are also placed in `data/raw_data/` (see paper for references).
- **Preprocessing** (e.g., normalization and windowing) is automatically handled by the code.

---

## Usage Examples

### Generate Synthetic Data

```bash
python3 main.py --data_name=noisy_sine --model=Diffusion --mode=generate --column=noise_scale_00 --representation=ts
```

### Benchmark Generated Data

```bash
python3 main.py --data_name=noisy_sine --model=Diffusion --mode=benchmark --column=noise_scale_00 --representation=ts
```

To see all available options:

```bash
python main.py --help
```

---

## Supported Models

- **WGAN_GP** — Wasserstein GAN with Gradient Penalty
- **TimeGAN** — Temporal Generative Adversarial Network
- **Diffusion** — Score-based generative model

---

## Evaluation Metrics

- **Predictive Score** — Generalization of models trained on synthetic data to real data
- **Discriminative Score** — Distinguishability of real vs synthetic data
- **Augmentation Score** — Performance gain from training on synthetic + real data
- **UMAP** — Dimensionality reduction for visual quality checks

---

## Results Logging

Benchmark results are logged to a CSV file using the built-in `write_results` utility.

### Location

```
results/backtest/scores.csv
```

### Fields

| Field           | Description                                      |
| --------------- | ------------------------------------------------ |
| model           | Generative model used                            |
| data_name       | Dataset name                                     |
| column          | Target column                                    |
| representation  | Representation method (e.g., `ts`, `xirp`)       |
| recovery_method | Method used to inverse transform representations |
| metric          | Evaluation metric name                           |
| score           | Mean score                                       |
| std             | Score standard deviation                         |
| datetime        | Timestamp of the run                             |

---

## Reproducing Tables and Figures in the Paper

| Paper Element               | Command / Script                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| Table 1 (benchmark metrics) | Run `main.py` with `--mode=benchmark` for each dataset & model. Results in `results/backtest/scores.csv`. |
| UMAP plots                  | UMAPs are generated during evaluation and stored in `results/umap/`.                                      |


---

## Data Sharing Policy

- No proprietary or NDA-restricted data is included.

---

## License & Use

The code is provided **solely for the purpose of reproducibility verification** for the manuscript submitted to the _International Journal of Forecasting_.

It is **not licensed for public distribution or commercial use**.  
For reuse beyond reproduction, please contact the authors for explicit permission.

---

## Help

To see all available flags and configuration options:

```bash
python main.py --help
```

---
