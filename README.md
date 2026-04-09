# Transformer-Based Jamming Detection

## Files
- `project/jammer_sim.py` - synthetic OFDM jammer data generator
- `project/models.py` - Transformer and MLP model definitions
- `project/train_single.py` - trains one model and writes metrics/results
- `requirements.txt` - Python dependencies

## Install
From the project root:

```bash
pip install -r requirements.txt
```

## How to run
The training script generates synthetic data on the fly, so no dataset materialization step is needed.

### Train the Transformer
From the project root:

```bash
cd project
python train_single.py --model transformer --seed 0 --output-dir ../outputs
```

### Train the MLP baseline
From the project root:

```bash
cd project
python train_single.py --model mlp --seed 0 --output-dir ../outputs
```

## Useful optional arguments
You can shorten a run for testing with smaller values:

```bash
cd project
python train_single.py --model transformer --seed 0 --output-dir ../outputs --train-samples 2000 --val-samples 500 --test-samples 500 --epochs 1 --batch-size 256
```

## What gets saved
The script writes results into the output directory you provide. For each run it saves:
- `*_history.csv` - training and validation metrics by epoch
- `*_pred.npy` - predicted test labels
- `*_true.npy` - ground-truth test labels
- `*_test_snr.npy` - SNR values for the test set
- `*_snr.csv` - SNR sweep accuracy / macro-F1
- `*_metrics.json` - final test metrics and confusion matrix

## Example output folder
After one run, you should see files like:

```text
outputs/
  transformer_seed0_history.csv
  transformer_seed0_pred.npy
  transformer_seed0_true.npy
  transformer_seed0_test_snr.npy
  transformer_seed0_snr.csv
  transformer_seed0_metrics.json
```
