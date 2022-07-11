import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.attention_drawer import drawer
from src.dataloader import get_data_X, get_data_y, get_loader
from src.model import CustomNet
from src.train import train

N_EPOCHS = 1000
DATA_SIZE = 1000
BATCH_SIZE = 1000
SEQ_LEN = 10
FEATURE_DIM = 5


def main():
    device = torch.device("cpu")

    model = CustomNet(
        input_size=5,
        seq_len=SEQ_LEN,
        num_hiddens=5,
        num_layers=2,
        dropout=0.1,
        attention_dim=10,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X = get_data_X(DATA_SIZE, SEQ_LEN, FEATURE_DIM, noise=0.001)
    y = get_data_y(DATA_SIZE)
    train_loader = get_loader(X, y, BATCH_SIZE)

    model, train_loss = train(
        model,
        N_EPOCHS,
        train_loader,
        optimizer,
        criterion,
        device,
    )


if __name__ == "__main__":
    main()
