# Bayesian-Mirror

This repository contains the code accompanying the paper 
> **Bayesian Mirror for False Discovery Rate Control.**

It includes the Bayesian mirror implementation, Python ports of the competing methods, and scripts that reproduce the simulation studies and the semi-synthetic study on real expression data.

## Files

| File | Contents |
|---|---|
| `libraries.py` | Third-party imports and version shims |
| `helpers.py` | Designs and data generation, FDP and power, knockoff-type threshold, early stopping, e-BH, plotting |
| `bayesian_mirror.py` | Bayesian mirror: cut-posterior draws, e-values, e-BH aggregation, multi-split procedure for high dimensions |
| `competing_methods.py` | Model-X and derandomised knockoffs, Gaussian mirror, data splitting and multiple data splitting |
| `low_dim_sim.py` | Low-dimensional study (n = 1000, p = 500) |
| `high_dim_sim.py` | High-dimensional study (n = 800, p = 2000) |
| `real_analysis.py` | Semi-synthetic study on the TCGA GBM expression design (n = 538, p = 12042) |
| `ablation.py` | Sensitivity of the Bayesian mirror to the screening fraction and the per-iterate level |

## Installation

```bash
pip install -r requirements.txt
```

## Example implementation

```python
import numpy as np
from bayesian_mirror import bayesian_mirror_multisplit

rng = np.random.default_rng(1884)
# X: n x p design, y: response
selected = bayesian_mirror_multisplit(X, y, q_star=0.1, gamma=0.03, T=1000, rng=rng,
   n_splits=20, split=0.4, early=True)
```

`selected` is a boolean mask over the p predictors.
- `q_star` is the target FDR level.
- `gamma` is the per-iterate level. It must satisfy `q_star > gamma / (1- gamma)`.
- `split` is the screening fraction n1/n.
- `n_splits` is the number of partitions L.


## Reproducing the simulations

```bash
python low_dim_sim.py
python high_dim_sim.py
```

Options:

- `--reps` sets the number of replications.
- `--n-jobs` sets the number of parallel workers.
- `--seed` sets the base seed.
- `--out-dir` sets the output folder.

Each run writes the per-replicate FDP and power to a CSV file and boxplots of FDP and power to each design.

## Reproducing the sensitivity analysis

```bash
python ablation.py
```

It reports empirical power and FDP for screening fractions 0.3 to 0.8 and per-iterate levels 0.02 to 0.08, on the AR(1) design with rho = 0.4 and s0 = 70. 

## Reproducing the semi-synthetic study

The design is the TCGA glioblastoma (GBM) gene expression data, Affymetrix HT_HG-U133A, Level 3 gene RMA, from the Broad Institute GDAC Firehose (https://gdac.broadinstitute.org). Download the file `GBM.transcriptome__ht_hg_u133a__broad_mit_edu__Level_3__gene_rma__data.data.txt` and run

```bash
python real_analysis.py
```

The options `--reps`, `-
