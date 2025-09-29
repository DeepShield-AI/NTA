# Algorithm Collection

This repository aggregates multiple network traffic classification projects. Each subproject has its own README with detailed usage. This top-level README provides a quick overview and entry points.

## Repository Structure

- `Fs-net/`
  - BiGRU-based sequence model (PyTorch + legacy TensorFlow) for encrypted traffic classification using packet-length sequences of length 256.
  - Key files: `fsnet_torch.py`, `train_torch.py`, `eval_torch.py`, `dataset_pcap_length.py`.
  - See `Fs-net/README.md` for details.

- `1d-CNN/`
  - PyTorch implementation of a 1D-CNN-style convolutional classifier trained on gzipped ubyte-format datasets (12/6/2-class).
  - Key files: `1d_cnn/cnn_1d_torch.py`, `1d_cnn/model.py`, `1d_cnn/data.py`.
  - See `1d-CNN/README.md` for details.

- `TFE-GNN/`
  - Official implementation of WWW'23 paper "TFE-GNN: A Temporal Fusion Encoder Using Graph Neural Networks for Fine-grained Encrypted Traffic Classification".
  - Provides full preprocessing from pcap to byte-level traffic graphs, training and evaluation scripts.
  - See `TFE-GNN/README.md` for environment setup and commands.

## Environments (summary)

- Fs-net (PyTorch path): Python 3.8+, PyTorch 1.9+, NumPy
- 1d-CNN (PyTorch): Python 3.8+, PyTorch 1.9+, torchvision, NumPy
- TFE-GNN (per subproject README): Python 3.8, specific Torch/DGL versions

Install examples (PyTorch minimal):

```bash
pip install torch torchvision torchaudio numpy
```

Refer to each subproject README for exact, version-pinned requirements.

## Data Expectations (high-level)

- `Fs-net/`
  - Uses text files with lines formatted as `<label> v1 v2 ... v256`.
  - Expected files: `train_pcap_length.txt`, `test_pcap_length.txt` in `Fs-net/`.

- `1d-CNN/`
  - Uses MNIST-style gzipped ubyte files per dataset folder: `*-images-idx3-ubyte.gz`, `*-labels-idx1-ubyte.gz` for both train and test.
  - Example folders under `1d-CNN/data/12class/`: `FlowAllLayers`, `FlowL7`, `SessionAllLayers`, `SessionL7`.

- `TFE-GNN/`
  - Starts from pcap files; includes scripts to split flows, convert to npz, and build byte-level graphs.
  - See `TFE-GNN/README.md` for download links and preprocessing steps.

## Quick Start

- Fs-net (PyTorch):
  ```bash
  cd Fs-net
  python train_torch.py              # train, saves to summary/
  python eval_torch.py               # evaluate latest checkpoint
  # or specify checkpoint
  python eval_torch.py --ckpt summary/model_100.pt
  ```

- 1d-CNN (PyTorch):
  ```bash
  cd 1d-CNN
  python 1d_cnn/cnn_1d_torch.py      # trains and evaluates, saves model.ckpt
  # To switch dataset folder or class count, edit cnn_1d_torch.py (folder_path_list, task_index, label_num)
  ```

- TFE-GNN:
  ```bash
  cd TFE-GNN
  # 1) Convert pcaps to npz
  python pcap2npy.py --dataset iscx-vpn
  # 2) Build graphs
  python preprocess.py --dataset iscx-vpn
  # 3) Train and test
  python train.py --dataset iscx-vpn --cuda 0
  python test.py  --dataset iscx-vpn --cuda 0
  ```

## Notes

- Each project is self-contained and may pin different dependency versions. Prefer using virtual environments per project.
- For reproducibility and stability, consider adding random seeds, input normalization, and regular checkpointing as needed.

## References

- TFE-GNN paper: https://dl.acm.org/doi/abs/10.1145/3543507.3583227
- Original 1d-CNN reference: https://github.com/mydre/wang-wei-s-research
