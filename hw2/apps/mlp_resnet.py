import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *(
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ),
        nn.Linear(hidden_dim, num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()
    error_count = 0
    num_samples = 0

    loss_sum = 0
    num_batches = 0
    f = nn.SoftmaxLoss()

    for batch_x, batch_y in dataloader:
        out = model(batch_x)
        error_count += (np.argmax(out.numpy(), axis=1) != batch_y.numpy()).sum()
        num_samples += batch_x.shape[0]
        loss = f(out, batch_y)
        num_batches += 1
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy()

    return error_count / num_samples, loss_sum / num_batches
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_error = 0
    training_loss = 0
    for _ in range(epochs):
        training_error, training_loss = epoch(train_dataloader, model, opt)

    model.eval()
    test_error, test_loss = epoch(test_dataloader, model)
    return training_error, training_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
