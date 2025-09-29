# Fs-net
FS-Net: A Flow Sequence Network for Encrypted Traffic Classification

This repository contains an implementation of FS-Net for encrypted traffic classification using sequence models and a dataset pipeline based on packet-length sequences. While the current PyTorch model is a BiGRU-based sequence encoder–decoder, this README analyzes the project from a 1D-CNN perspective and provides guidance to build a 1D-CNN baseline on the existing data pipeline.

## Project overview

- Current primary model (PyTorch): `fsnet_torch.py` defines `FsNetTorch`, a 2-layer bidirectional GRU encoder and a 2-layer bidirectional GRU decoder with a classification head. Input is a univariate sequence of length 256, output is 4 classes.
- Legacy TensorFlow model: `model.py` defines a TF1.x variant (`Fs_net`) with similar encoder/decoder design and SELU-activated dense layers, trained via `train.py`.
- Data pipeline: `dataset_pcap_length.py` reads lines of "label + 256 integer values" as a sequence, producing tensors shaped `(batch, 256, 1)`.

From the 1D-CNN perspective, this dataset is a classic 1D time-series classification problem with:
- Sequence length: 256
- Channels (features per step): 1
- Classes: 4 (`iqiyi`, `taobao`, `weibo`, `weixin`)

## Repository structure

- `fsnet_torch.py`: PyTorch model (BiGRU encoder/decoder + classifier)
- `train_torch.py`: PyTorch training loop on `dataset_pcap_length.py`
- `eval_torch.py`: PyTorch evaluation of saved checkpoints
- `dataset_pcap_length.py`: Dataset provider for 4-class packet-length sequences
- `traffic_dataset.py`: Alternative dataset (NIMS_*.arff) with 11 classes and length 22 (unused by default)
- `train.py`: Legacy TF1.x training script for `Fs_net`
- `model.py`: Legacy TF1.x model definition
- `graph.png`, `tensorboard.png`: Illustrations/plots
- Expected data files: `train_pcap_length.txt`, `test_pcap_length.txt`

## Data format

`dataset_pcap_length.py` expects lines like:

```
<label> v1 v2 v3 ... v256
```

- Labels are mapped as `{"iqiyi": 0, "taobao": 1, "weibo": 2, "weixin": 3}`.
- Each `vi` is an integer packet-length value. The loader converts to float and shapes data as `(batch, 256, 1)`.

For the alternative `traffic_dataset.py` (NIMS dataset):
- Lines are comma-separated with the last field being the label string among 11 classes, sequence length 22. This path is disabled by default in `train.py`.

## Environment

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- Optional (legacy): TensorFlow 1.x if you intend to run the old `train.py`

Install minimal dependencies for PyTorch usage:

```bash
pip install torch torchvision torchaudio numpy
```

## Usage (current BiGRU model)

### Train

```bash
python train_torch.py \
  --n_steps 256 \
  --n_inputs 1 \
  --n_outputs 4 \
  --batch_size 8 \
  --lr 1e-3 \
  --n_epoch 1500 \
  --log_every_steps 100 \
  --model_dir summary/
```

Checkpoints are saved every 2 epochs to `summary/model_<epoch>.pt`.

### Evaluate

Evaluate the latest checkpoint in `summary/`:

```bash
python eval_torch.py
```

Or specify a particular checkpoint file:

```bash
python eval_torch.py --ckpt summary/model_100.pt
```