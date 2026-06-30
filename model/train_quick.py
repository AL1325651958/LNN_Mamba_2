"""
Quick training script — optimized for speed.
Runs 20 epochs on a subset of Wind Farm Site 2 data.
"""
import sys, os, io, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from model import LNNMambaTransformer, WindDataModule


def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy().ravel()
    targ_np = target.detach().cpu().numpy().ravel()
    mask = targ_np > 0
    if mask.sum() > 0:
        pred_np, targ_np = pred_np[mask], targ_np[mask]
    mse = np.mean((pred_np - targ_np) ** 2)
    mae = np.mean(np.abs(pred_np - targ_np))
    eps = 1e-4
    mape = np.mean(np.abs((targ_np - pred_np) / (targ_np + eps))) * 100
    return {'rmse': np.sqrt(mse), 'mae': mae, 'mape': mape, 'mse': mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=48)
    parser.add_argument('--n_mamba', type=int, default=1)
    parser.add_argument('--n_tf', type=int, default=1)
    parser.add_argument('--d_state', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--train_ratio', type=float, default=0.15)
    parser.add_argument('--n_vars_use', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | PyTorch {torch.__version__}')

    # Load limited data
    print('\n--- Loading Data ---')
    dm = WindDataModule(
        site_name='Wind_farm_site_2_200MW',
        seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, stride=4,  # stride=4 reduces samples
        train_ratio=args.train_ratio, val_ratio=0.15,
    )
    train_loader, val_loader, test_loader = dm.get_dataloaders()
    print(f'Train batches: {len(train_loader)}, Val: {len(val_loader)}')

    # Build model
    print('\n--- Building Model ---')
    model = LNNMambaTransformer(
        n_vars=dm.n_vars,
        d_model=args.d_model,
        n_mamba_blocks=args.n_mamba,
        n_transformer_layers=args.n_tf,
        d_state=args.d_state,
        d_conv=3,
        n_heads=4,
        pred_len=args.pred_len,
        lnn_hidden=24,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if device.type == 'cuda' else None

    # Training
    os.makedirs('checkpoints', exist_ok=True)
    best_val_rmse = float('inf')
    history = {'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_mape': [], 'epoch_time': []}

    print(f'\n{"="*60}')
    print(f'TRAINING: {args.epochs} epochs, seq={args.seq_len}, pred={args.pred_len}')
    print(f'd_model={args.d_model}, mamba_blocks={args.n_mamba}, d_state={args.d_state}')
    print(f'Batches: {len(train_loader)}/epoch')
    print(f'{"="*60}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        for batch_idx, (x, y, ts) in enumerate(train_loader):
            x, y, ts = x.to(device), y.to(device), ts.to(device)
            optimizer.zero_grad()

            with autocast(enabled=scaler is not None):
                out = model(x, ts)
                loss_dict = model.compute_loss(out['pred'], y, out['regime'])
                loss = loss_dict['loss']

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y, ts in val_loader:
                x, y, ts = x.to(device), y.to(device), ts.to(device)
                out = model(x, ts)
                all_preds.append(out['pred'].cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds).ravel()
        targets = np.concatenate(all_targets).ravel()

        # Inverse transform
        if dm.scaler_power is not None:
            preds = dm.scaler_power.inverse_transform(preds.reshape(-1, 1)).ravel()
            targets = dm.scaler_power.inverse_transform(targets.reshape(-1, 1)).ravel()

        metrics = compute_metrics(torch.tensor(preds), torch.tensor(targets))
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_rmse'].append(metrics['rmse'])
        history['val_mae'].append(metrics['mae'])
        history['val_mape'].append(metrics['mape'])
        history['epoch_time'].append(elapsed)

        print(f'Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | '
              f'Val RMSE: {metrics["rmse"]:.2f} MW | MAE: {metrics["mae"]:.2f} | '
              f'MAPE: {metrics["mape"]:.1f}% | {elapsed:.0f}s | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Save best
        if metrics['rmse'] < best_val_rmse:
            best_val_rmse = metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_rmse': best_val_rmse,
                'history': history,
            }, f'checkpoints/lmt_quick_{args.train_ratio}.pt')
            print(f'  ★ Best model (RMSE={best_val_rmse:.2f} MW)')

    # --- Test ---
    print(f'\n{"="*60}')
    print('FINAL TEST EVALUATION')
    ckpt = torch.load(f'checkpoints/lmt_quick_{args.train_ratio}.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y, ts in test_loader:
            x, y, ts = x.to(device), y.to(device), ts.to(device)
            out = model(x, ts)
            all_preds.append(out['pred'].cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()
    if dm.scaler_power is not None:
        preds = dm.scaler_power.inverse_transform(preds.reshape(-1, 1)).ravel()
        targets = dm.scaler_power.inverse_transform(targets.reshape(-1, 1)).ravel()

    test_m = compute_metrics(torch.tensor(preds), torch.tensor(targets))
    print(f'Test RMSE: {test_m["rmse"]:.2f} MW')
    print(f'Test MAE:  {test_m["mae"]:.2f} MW')
    print(f'Test MAPE: {test_m["mape"]:.1f}%')
    print(f'Params:    {n_params:,}')
    print(f'Best Val:  {best_val_rmse:.2f} MW')

    # Per-horizon
    pred_h = preds.reshape(-1, args.pred_len)
    targ_h = targets.reshape(-1, args.pred_len)
    print(f'\nPer-horizon RMSE (MW):')
    for h in range(min(args.pred_len, 12)):
        rmse_h = np.sqrt(np.mean((pred_h[:, h] - targ_h[:, h]) ** 2))
        print(f'  t+{(h+1)*15:3d}min: {rmse_h:.2f}')

    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_m.items()},
        'best_val_rmse': float(best_val_rmse),
        'n_params': n_params,
        'history': {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open('results_lmt_quick.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to results_lmt_quick.json')


if __name__ == '__main__':
    main()
