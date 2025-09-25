# coding=utf-8
import argparse
import glob
import os
import re
from typing import Optional

import torch
import torch.nn as nn

from fsnet_torch import FsNetTorch
from dataset_pcap_length import dataset


def find_latest_checkpoint(summary_dir: str) -> Optional[str]:
    pattern = os.path.join(summary_dir, 'model_*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    def epoch_num(p: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_num, reverse=True)
    return files[0]


def evaluate(model: nn.Module, test_data, batch_size: int, device: torch.device) -> float:
    model.eval()
    steps_per_epoch = test_data.num_examples // batch_size
    correct_cnt = 0
    num_test_examples = steps_per_epoch * batch_size
    with torch.no_grad():
        for _ in range(steps_per_epoch):
            X_test_np, y_test_np = test_data.next_batch(batch_size)
            X_test = torch.from_numpy(X_test_np).float().to(device)
            y_test = torch.from_numpy(y_test_np).long().to(device)
            logits = model(X_test)
            pred = torch.argmax(logits, dim=1)
            correct_cnt += (pred == y_test).sum().item()
    return (correct_cnt * 1.0 / num_test_examples) if num_test_examples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Evaluate FsNetTorch checkpoint on test set')
    parser.add_argument('--ckpt', type=str, default='', help='Path to checkpoint .pt (from train_torch.py)')
    parser.add_argument('--summary_dir', type=str, default='summary/', help='Directory containing model_*.pt files')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_data = dataset.test()

    # Build model with same hyperparameters as training
    model = FsNetTorch(
        n_steps=256,
        n_inputs=1,
        n_outputs=4,
        n_neurons=128,
        encoder_n_neurons=128,
        decoder_n_neurons=128,
    ).to(device)

    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.summary_dir) or ''
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError("Checkpoint not found. Given: '{}' . Searched latest in '{}' .".format(args.ckpt, args.summary_dir))

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise RuntimeError("Unexpected checkpoint format. Expected dict with 'model_state_dict'.")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    acc = evaluate(model, test_data, args.batch_size, device)
    print("Loaded: {}".format(ckpt_path))
    print("Test accuracy: {:.6f}".format(acc))


if __name__ == '__main__':
    main()
