"""
Training script for LNN-Gated Selective SSM-Transformer (LMT) wind power forecasting.

Usage:
    python -m model.train --site Wind_farm_site_2_200MW --epochs 100 --batch_size 32

Features:
  - Mixed precision training (AMP)
  - Gradient clipping
  - Cosine annealing + warmup
  - Early stopping
  - TensorBoard logging
  - Evaluation metrics (RMSE, MAE, MAPE, R²)
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LNNMambaTransformer, WindDataModule, get_available_sites


# ============================================================
# Training Utilities
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        model.load_state_dict(self.best_state)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    mask = target > 0  # only evaluate on non-zero ground truth
    if mask.sum() > 0:
        pred_f = pred[mask]
        target_f = target[mask]
    else:
        pred_f = pred
        target_f = target

    mse = np.mean((pred_f - target_f) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_f - target_f))

    # MAPE with epsilon to avoid div by zero
    eps = 1e-4
    mape = np.mean(np.abs((target_f - pred_f) / (target_f + eps))) * 100

    # R²
    ss_res = np.sum((target_f - pred_f) ** 2)
    ss_tot = np.sum((target_f - np.mean(target_f)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
    }


# ============================================================
# Training Loop
# ============================================================

def train_epoch(model, loader, optimizer, scaler, device, lambda_regime=0.1):
    model.train()
    losses = []
    mse_list = []

    for batch_idx, (x, y, ts) in enumerate(loader):
        x, y, ts = x.to(device), y.to(device), ts.to(device)

        optimizer.zero_grad()

        with autocast(enabled=scaler is not None):
            out = model(x, ts, return_aux=False)
            loss_dict = model.compute_loss(out['pred'], y, out['regime'],
                                           lambda_regime=lambda_regime)

        loss = loss_dict['loss']

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.item())
        mse_list.append(loss_dict['mse'].item())

        if batch_idx % 50 == 0:
            print(f'  Batch {batch_idx}: loss={loss.item():.4f}, rmse={loss_dict["rmse"].item():.4f}')

    return np.mean(losses), np.mean(mse_list)


@torch.no_grad()
def evaluate(model, loader, scaler_power, device):
    model.eval()
    all_preds = []
    all_targets = []

    for x, y, ts in loader:
        x, y, ts = x.to(device), y.to(device), ts.to(device)
        out = model(x, ts, return_aux=False)
        pred = out['pred']

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Flatten
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    # Inverse transform to original power scale
    if scaler_power is not None:
        preds_flat = scaler_power.inverse_transform(preds_flat.reshape(-1, 1)).ravel()
        targets_flat = scaler_power.inverse_transform(targets_flat.reshape(-1, 1)).ravel()

    metrics = compute_metrics(preds_flat, targets_flat)
    return metrics, preds_flat, targets_flat


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Check available sites
    print('\nAvailable wind farm sites:')
    for site in get_available_sites():
        print(f'  {site["name"]}: {site["total_rows"]} rows, {len(site["files"])} segment(s)')

    # Data
    data_module = WindDataModule(
        data_dir=args.data_dir,
        site_name=args.site,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    # Model
    model = LNNMambaTransformer(
        n_vars=data_module.n_vars,
        d_model=args.d_model,
        n_mamba_blocks=args.n_mamba_blocks,
        n_transformer_layers=args.n_transformer_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        pred_len=args.pred_len,
        lnn_hidden=args.lnn_hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel parameters: {n_params:,}')

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.warmup_epochs + 10, T_mult=2, eta_min=args.lr * 0.01,
    )

    # Mixed precision
    scaler = GradScaler() if device.type == 'cuda' else None

    # Early stopping
    early_stop = EarlyStopping(patience=args.patience)

    # Logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/{args.site}_{timestamp}'
    writer = SummaryWriter(log_dir)
    os.makedirs('checkpoints', exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Training started — log: {log_dir}')
    print(f'seq_len={args.seq_len}, pred_len={args.pred_len}, batch={args.batch_size}')
    print(f'd_model={args.d_model}, mamba_blocks={args.n_mamba_blocks}, '
          f'transformer_layers={args.n_transformer_layers}')
    print(f'{"="*60}\n')

    best_val_rmse = float('inf')
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_mse = train_epoch(model, train_loader, optimizer, scaler, device)

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, data_module.scaler_power, device)
        val_rmse = val_metrics['rmse']

        # Scheduler
        scheduler.step()

        # Logging
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
        writer.add_scalar('MAPE/val', val_metrics['mape'], epoch)
        writer.add_scalar('LR', lr, epoch)

        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_metrics['mae'])
        history['val_mape'].append(val_metrics['mape'])

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Val RMSE: {val_rmse:.4f} MW | MAE: {val_metrics["mae"]:.4f} | '
              f'MAPE: {val_metrics["mape"]:.2f}% | R²: {val_metrics["r2"]:.3f} | '
              f'LR: {lr:.2e} | {elapsed:.1f}s')

        # Checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'history': dict(history),
                'args': vars(args),
            }, f'checkpoints/best_{args.site}.pt')
            print(f'  ✓ Best model saved (RMSE={val_rmse:.4f})')

        # Early stopping
        if early_stop(val_rmse, model):
            print(f'\nEarly stopping at epoch {epoch}')
            early_stop.restore(model)
            break

    writer.close()

    # ============================================================
    # Final Evaluation
    # ============================================================
    print(f'\n{"="*60}')
    print('FINAL EVALUATION')

    checkpoint = torch.load(f'checkpoints/best_{args.site}.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded best model from epoch {checkpoint["epoch"]} (val RMSE={checkpoint["val_rmse"]:.4f})')

    # Test
    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, data_module.scaler_power, device
    )

    # Multi-horizon evaluation
    print(f'\nTest Metrics (all horizons):')
    print(f'  RMSE:  {test_metrics["rmse"]:.4f} MW')
    print(f'  MAE:   {test_metrics["mae"]:.4f} MW')
    print(f'  MAPE:  {test_metrics["mape"]:.2f}%')
    print(f'  R²:    {test_metrics["r2"]:.4f}')

    # Per-horizon metrics
    print(f'\nPer-Horizon RMSE (MW):')
    pred_horizon = test_preds.reshape(-1, args.pred_len)
    target_horizon = test_targets.reshape(-1, args.pred_len)

    for h in [0, 3, 5, 11, 23, 47, 95]:
        if h < args.pred_len:
            rmse_h = np.sqrt(np.mean((pred_horizon[:, h] - target_horizon[:, h]) ** 2))
            mae_h = np.mean(np.abs(pred_horizon[:, h] - target_horizon[:, h]))
            print(f'  t+{(h+1)*15}min:  RMSE={rmse_h:.4f}, MAE={mae_h:.4f}')

    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_val_rmse': float(best_val_rmse),
        'n_params': n_params,
        'args': vars(args),
    }
    results_path = f'results_{args.site}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')

    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LNN-Gated Selective SSM-Transformer')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/wind')
    parser.add_argument('--site', type=str, default='Wind_farm_site_2_200MW')
    parser.add_argument('--seq_len', type=int, default=336, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_mamba_blocks', type=int, default=3)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--lnn_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    args = parser.parse_args()
    main(args)
