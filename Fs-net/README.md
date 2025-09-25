# Fs-net
FS-Net: A Flow Sequence Network For Encrypted Traffic Classification

This is an implementation about FS-net
 
## PyTorch implementation

Files:
- `fsnet_torch.py`: PyTorch model definition (2-layer BiGRU encoder/decoder + classifier).
- `train_torch.py`: Training loop using the existing dataset providers in `dataset_pcap_length.py`.
- `eval_torch.py`: Evaluate saved checkpoints on the test set.

Data files expected in this folder:
- `train_pcap_length.txt`
- `test_pcap_length.txt`

### Train

```bash
python train_torch.py
```

This will create checkpoints under `summary/model_<epoch>.pt` every 2 epochs.

### Evaluate

Evaluate the latest checkpoint in `summary/`:

```bash
python eval_torch.py
```

Or specify a particular checkpoint file:

```bash
python eval_torch.py --ckpt summary/model_100.pt
```

Notes:
- The training loop mirrors the TensorFlow script `train.py` (batch size 8, Adam lr=1e-3, 1500 epochs, periodic logging). L2 regularization is applied only to classifier weights (`fc1.weight`, `fc2.weight`) with weight decay 3e-3 to match the TF dense-layer L2 regularizer.