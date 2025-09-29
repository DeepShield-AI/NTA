# 1D-CNN Traffic Classification

This project provides a PyTorch implementation of a 1D-CNN-like convolutional classifier for network traffic representation images/sequences. It loads gzipped ubyte-formatted datasets and trains a compact convolutional model for multi-class classification (commonly 12 classes: Novpn/Vpn variants across application categories).

The repository also includes an unmaintained TensorFlow reference (`1d_cnn/cnn_1d_tensorflow.py`). The recommended path is the PyTorch implementation.

## Repository Structure

- `1d_cnn/cnn_1d_torch.py`: Main PyTorch training/evaluation script (hard-coded hyperparameters and paths).
- `1d_cnn/model.py`: CNN architectures (`OneCNN`, `OneCNNC`, `CNNImage`). Torch script uses `OneCNNC` by default.
- `1d_cnn/data.py`: Data loader for gzipped ubyte-format datasets (`DealDataset`).
- `1d_cnn/test_data.py`: Minimal gzip reading snippet for debugging.
- `1d_cnn/cnn_1d_tensorflow.py`: Original TensorFlow script (for reference only).
- `data/12class/…`: Dataset folders (e.g., `FlowAllLayers`, `FlowL7`, `SessionAllLayers`, `SessionL7`).
- `README.md`: This document.

## Environment

- Python >= 3.8
- PyTorch >= 1.9
- torchvision
- numpy

Install minimal dependencies:

```bash
pip install torch torchvision numpy
```

## Data Format

`1d_cnn/data.py` expects MNIST-style gzipped ubyte files in each dataset folder:

- Images: `*-images-idx3-ubyte.gz`
- Labels: `*-labels-idx1-ubyte.gz`

The loader reads images to shape `(N, 28, 28)` and labels to `(N,)`. In `DealDataset.__getitem__`, images are reshaped to `(1, 1, 784)` to be consumed by a 2D convolution with kernel size `(1, 25)` (i.e., a 1D convolution along the flattened temporal dimension).

Dataset folders provided under `data/12class/` (12-class setting):

- `FlowAllLayers`
- `FlowL7`
- `SessionAllLayers`
- `SessionL7`

Each folder should contain the four files:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

## Classification Tasks

- 2-class: `Novpn`, `Vpn`
- 6-class: `Chat`, `Email`, `File`, `P2p`, `Streaming`, `Voip`
- 12-class: `Chat`, `Email`, `File`, `P2p`, `Streaming`, `Voip`, `Vpn_Chat`, `Vpn_Email`, `Vpn_File`, `Vpn_P2p`, `Vpn_Streaming`, `Vpn_Voip`

The PyTorch script is preconfigured for 12-class classification. To use 2-class or 6-class, ensure the corresponding dataset and labels exist and adjust `label_num` accordingly.

## Training and Evaluation (PyTorch)

Run the training script:

```bash
python 1d_cnn/cnn_1d_torch.py
```