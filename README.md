# Leveraging Image-based Generative Adversarial Networks For Time Series Generation

This repository provides the code supplementing Leveraging Image-based Generative Adversarial
Networks For Time Series Generation by Hellermann and Lessmann.

## Contact

## For questions or support, please contact **Justin Hellermann** at [justin.hellermann@hu-berlin.com](mailto:justin.hellermann@hu-berlin.com).

## Environment Setup

IMPORTANT: Each model has specific Python version compatibility!

| Model     | Python Version | Requirements File               |
| --------- | -------------- | ------------------------------- |
| WGAN_GP   | Python ≥ 3.8   | requirements_wgan_diffusion.txt |
| Diffusion | Python ≥ 3.8   | requirements_wgan_diffusion.txt |
| TimeGAN   | Python ≤ 3.6   | requirements_timegan.txt        |

---

### Operating System Compatibility

This project is intended to run in a POSIX-compliant environment, such as Linux. It has been tested on systems reporting the following, when execeuted in Python:

- `os.name`: `posix`
- `platform.system()`: `Linux`
- `platform.release()`: `6.1.x` (or similar)

Ensure your environment supports standard Unix-like behavior for full compatibility with shell commands and system dependencies.

## Installation

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

## Usage

### Generate Synthetic Data

```bash
python3 main.py --data_name=noisy_sine --model=Diffusion --mode=generate --column=noise_scale_00 --representation=ts
```

### Benchmark Generated Data

After running the code with the generation flag, the benchmark flag will conduct the benchmarking. Make sure all combinations have been run before calling the benchmark script. Since the comparison is only conducted among trials, which have successfully run.

```bash
python3 main.py --data_name=noisy_sine --model=Diffusion --mode=benchmark --column=noise_scale_00 --representation=ts
```

---

## Supported Models

- WGAN_GP: Wasserstein GAN with Gradient Penalty
- TimeGAN: Temporal Generative Adversarial Network
- Diffusion: Score-based generative model

---

## Evaluation Metrics

The script supports benchmarking using the following metrics:

- Predictive Score — assesses how well a model trained on synthetic data generalizes to real data.
- Discriminative Score — evaluates whether a classifier can distinguish real from synthetic data.
- Augmentation Score — tests the utility of synthetic data in boosting performance when used with real data.

UMAP and t-SNE visualizations are supported for qualitative comparisons. In the results folder, you can find a selection of UMAP plots, since uploading all plots exceeds the capacity of the portal.

---

## Data & Output

- Input CSV files should be placed in:  
  `data/raw_data/{data_name}.csv`

- Output files (generated and original sequences) are saved to:
  - `data/syn/` — synthetic sequences
  - `data/ori/` — original sequences

---

## Results Logging

All benchmark results (metrics) are automatically logged to a CSV file using the `write_results` utility function.

### Location

```
results/backtest/scores.csv
```

### Logged Fields

Each row in the file includes:

| Field           | Description                                     |
| --------------- | ----------------------------------------------- |
| model           | The generative model used                       |
| data_name       | Dataset name                                    |
| column          | Target column used                              |
| representation  | Representation method (e.g., xirp, gasf)        |
| recovery_method | Recovery method used for inverse transformation |
| metric          | Metric name (e.g., pred_score, disc_score)      |
| score           | Mean score value                                |
| std             | Standard deviation of the score                 |
| datetime        | Timestamp of the run                            |

If the CSV file doesn't exist yet, the header row is created. All subsequent runs append to the file.

---

## Help

To explore all available arguments:

```bash
python main.py --help
```

## License & Use

The code is provided solely to support the reproducibility check of our paper submitted to the International Journal of Forecasting.  
It is not licensed for public distribution or commercial use.  
Please contact the authors for reuse permissions.
