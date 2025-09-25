# coding=utf-8
import os
import random
import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fsnet_torch import FsNetTorch
from dataset_pcap_length import dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser = argparse.ArgumentParser(description="Train FsNetTorch to mimic TensorFlow logging behavior")
    parser.add_argument('--n_steps', type=int, default=256)
    parser.add_argument('--n_inputs', type=int, default=1)
    parser.add_argument('--n_outputs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epoch', type=int, default=1500)
    # Match historical log style: print train_accuracy every 100 steps by default
    parser.add_argument('--log_every_steps', type=int, default=100)
    parser.add_argument('--model_dir', type=str, default='./summary/')
    # By default, do NOT print test accuracy each epoch to mimic the log that shows mostly train_accuracy
    parser.add_argument('--print_test_each_epoch', action='store_true', help='Print test accuracy after each epoch')
    args = parser.parse_args()

    # Hyperparameters analogous to TensorFlow train.py (overridable via args)
    n_steps = args.n_steps
    n_inputs = args.n_inputs
    n_outputs = args.n_outputs
    batch_size = args.batch_size
    lr = args.lr
    n_epoch = args.n_epoch
    log_every_steps = args.log_every_steps
    model_dir = args.model_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    os.makedirs(model_dir, exist_ok=True)

    # Data providers
    train_data = dataset.train()
    test_data = dataset.test()

    # Model
    model = FsNetTorch(
        n_steps=n_steps,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_neurons=128,
        encoder_n_neurons=128,
        decoder_n_neurons=128,
    ).to(device)

    # Loss and optimizer (L2 on classifier weights only to mimic TF regularizer on dense layers)
    criterion = nn.CrossEntropyLoss()
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if name in ["fc1.weight", "fc2.weight"]:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    optimizer = optim.Adam([
        {"params": decay_params, "weight_decay": 3e-3},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)

    # Training loop
    cnt = 0
    for epoch in range(n_epoch):
        model.train()
        batch_num = train_data.num_examples // batch_size
        for _ in range(batch_num):
            X_batch_np, y_batch_np = train_data.next_batch(batch_size)
            X_batch = torch.from_numpy(X_batch_np).float().to(device)
            y_batch = torch.from_numpy(y_batch_np).long().to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            cnt += 1
            if cnt % log_every_steps == 0:
                with torch.no_grad():
                    train_acc = (torch.argmax(logits, dim=1) == y_batch).float().mean().item()
                # Match historical log line format (no fixed decimals)
                print("step: {} train_accuracy: {}".format(cnt, train_acc), flush=True)

        # Optionally print test accuracy once per epoch
        if args.print_test_each_epoch:
            acc = evaluate(model, test_data, batch_size, device)
            print("step: {} test_accuracy: {}".format(cnt, acc), flush=True)

        # Save checkpoint every 2 epochs (similar to TF saver frequency)
        if epoch % 2 == 0:
            ckpt_path = os.path.join(model_dir, "model_{}.pt".format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)


if __name__ == "__main__":
    main()