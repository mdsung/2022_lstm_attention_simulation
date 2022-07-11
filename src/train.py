import numpy as np

from src.attention_drawer import drawer


def train(
    model,
    n_epochs,
    train_loader,
    optimizer,
    criterion,
    device,
):
    # initialize lists to monitor train, valid loss and accuracy
    train_losses = []
    avg_train_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            output, _ = model(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
        )
        print(print_msg)

        if epoch % 100 == 0:
            drawer(epoch, model)

        # clear lists to track next epoch
        train_losses = []

    return (
        model,
        avg_train_losses,
    )
