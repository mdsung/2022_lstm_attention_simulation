import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data_X(
    data_size: int, seq_length: int, feature_dim: int, noise: float = 0.01
) -> torch.Tensor:
    early_arr = np.ones((data_size, seq_length // 2, feature_dim))
    late_arr1 = np.ones((data_size, (seq_length // 2), feature_dim))
    late_arr2 = np.zeros((data_size, (seq_length // 2), feature_dim))

    label1 = (
        np.concatenate([early_arr, late_arr1], axis=1)
        + np.random.randn(data_size, seq_length, feature_dim) * noise
    )

    label2 = (
        np.concatenate([early_arr, late_arr2], axis=1)
        + np.random.randn(data_size, seq_length, feature_dim) * noise
    )

    return torch.from_numpy(np.concatenate([label1, label2], axis=0)).float()


def get_data_y(data_size: int) -> torch.Tensor:
    return torch.vstack(
        (
            torch.vstack([torch.Tensor([1, 0])] * data_size),
            torch.vstack([torch.Tensor([0, 1])] * data_size),
        )
    )


def get_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(X, y), batch_size=batch_size, shuffle=True, num_workers=0
    )


if __name__ == "__main__":
    X = get_data_X(1000, 10, 5)
    y = get_data_y(1000)

    for X, y in get_loader(X, y, 10):
        print(X)
        print(y)
        break
