"""
Data pipeline for wind power forecasting.
Reads clean CSV files, normalizes, and creates sliding windows.

Supports:
  - Multi-variable input (wind speed × heights, direction, temp, pressure, humidity)
  - Time features (hour, month, day-of-year)
  - Train/val/test split by time
  - Sliding window generation
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import glob
from typing import Optional, Tuple, List, Dict


class WindFarmDataset(Dataset):
    """
    Wind Farm multivariate time series dataset.

    Each sample: (x_history, y_future, timestamps, metadata)
      x_history:  (n_vars, seq_len)  input window
      y_future:   (pred_len,)         target power values
    """

    def __init__(
        self,
        data: np.ndarray,
        time_features: np.ndarray,
        seq_len: int = 336,
        pred_len: int = 96,
        stride: int = 1,
    ):
        """
        Args:
            data:          (T, V)  multivariate time series
            time_features: (T, 2)  [hour, month] features
            seq_len:       input sequence length
            pred_len:      prediction horizon
            stride:        sliding window stride
        """
        self.data = torch.FloatTensor(data)  # (T, V)
        self.time_features = torch.FloatTensor(time_features)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

        self.n_samples = max(0, (len(data) - seq_len - pred_len) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end_in = start + self.seq_len
        end_out = end_in + self.pred_len

        x = self.data[start:end_in]          # (seq_len, V)
        y = self.data[end_in:end_out, -1]    # (pred_len,) Power column = last
        ts = self.time_features[start:end_in]

        # Transpose to (V, seq_len) for model input
        x = x.transpose(0, 1)  # (V, seq_len)

        return x, y, ts


class WindDataModule:
    """
    Data module managing train/val/test splits for wind farm forecasting.
    Reads clean CSV files from data/wind/ directory.
    """

    def __init__(
        self,
        data_dir: str = 'data/wind',
        site_name: str = 'Wind_farm_site_2_200MW',
        seq_len: int = 336,
        pred_len: int = 96,
        batch_size: int = 32,
        stride: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: str = 'standard',
    ):
        self.data_dir = data_dir
        self.site_name = site_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler_type = scaler_type

        self.scaler = None
        self.scaler_power = None
        self.n_vars = None
        self.var_names = None

    def load_and_preprocess(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load data, fit scalers, create datasets."""

        # Find site CSV files
        pattern = os.path.join(self.data_dir, f'{self.site_name}*.csv')
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            raise FileNotFoundError(f'No CSV files found: {pattern}')

        print(f'Loading {len(csv_files)} segment(s) for {self.site_name}')

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f'  {os.path.basename(f)}: {len(df)} rows')

        df = pd.concat(dfs, ignore_index=True)

        # Parse time column
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        print(f'Total rows: {len(df)}, Time range: {df[time_col].min()} ~ {df[time_col].max()}')

        # Extract variable columns (skip time)
        val_cols = [c for c in df.columns if c != time_col]
        self.var_names = val_cols

        # Separate Power column (last) from features
        feature_cols = val_cols[:-1]
        power_col = val_cols[-1]
        print(f'Features ({len(feature_cols)}): {feature_cols}')
        print(f'Target: {power_col}')

        # Extract time features
        hours = df[time_col].dt.hour.values
        months = df[time_col].dt.month.values
        doys = df[time_col].dt.dayofyear.values
        time_feat = np.stack([hours, months], axis=1)  # (T, 2)

        # Feature data
        features = df[feature_cols].values.astype(np.float32)
        power = df[power_col].values.astype(np.float32).reshape(-1, 1)

        # Scale features
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        features_scaled = self.scaler.fit_transform(features)

        # Scale power separately (we compute metrics on original scale)
        if self.scaler_type == 'standard':
            self.scaler_power = StandardScaler()
        else:
            self.scaler_power = RobustScaler()
        power_scaled = self.scaler_power.fit_transform(power)

        # Combine: features + power
        data = np.concatenate([features_scaled, power_scaled], axis=1)  # (T, V)
        self.n_vars = data.shape[1]

        # Split by time
        T = len(data)
        train_end = int(T * self.train_ratio)
        val_end = train_end + int(T * self.val_ratio)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_time = time_feat[:train_end]
        val_time = time_feat[train_end:val_end]
        test_time = time_feat[val_end:]

        print(f'\nSplit: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}')
        print(f'Train: {df[time_col].iloc[0]} ~ {df[time_col].iloc[train_end-1]}')
        print(f'Val:   {df[time_col].iloc[train_end]} ~ {df[time_col].iloc[val_end-1]}')
        print(f'Test:  {df[time_col].iloc[val_end]} ~ {df[time_col].iloc[-1]}')

        self.train_dataset = WindFarmDataset(
            train_data, train_time, self.seq_len, self.pred_len, self.stride
        )
        self.val_dataset = WindFarmDataset(
            val_data, val_time, self.seq_len, self.pred_len, self.stride
        )
        self.test_dataset = WindFarmDataset(
            test_data, test_time, self.seq_len, self.pred_len, self.stride
        )

        print(f'Samples: train={len(self.train_dataset)}, val={len(self.val_dataset)}, '
              f'test={len(self.test_dataset)}')

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds, val_ds, test_ds = self.load_and_preprocess()
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader

    def inverse_transform_power(self, power_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled power predictions back to original scale."""
        return self.scaler_power.inverse_transform(power_scaled.reshape(-1, 1)).ravel()


def get_available_sites(data_dir: str = 'data/wind') -> List[Dict]:
    """List all available wind farm sites and their segment files."""
    sites = {}
    for f in glob.glob(os.path.join(data_dir, '*.csv')):
        # Extract site name from filename
        basename = os.path.basename(f)
        parts = basename.split('_seg')
        site_name = parts[0]

        if site_name not in sites:
            sites[site_name] = {'name': site_name, 'files': [], 'total_rows': 0}

        df = pd.read_csv(f, nrows=1)
        sites[site_name]['files'].append(basename)
        sites[site_name]['total_rows'] += len(pd.read_csv(f))

    return sorted(sites.values(), key=lambda x: -x['total_rows'])


if __name__ == '__main__':
    # Quick test
    mod = WindDataModule(seq_len=96, pred_len=24, batch_size=2)
    train, val, test = mod.get_dataloaders()
    x, y, ts = next(iter(train))
    print(f'\nSample batch: x={x.shape}, y={y.shape}, ts={ts.shape}')
    print(f'n_vars = {mod.n_vars}')
