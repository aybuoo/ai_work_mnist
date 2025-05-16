# train.py
import torch
from torch import nn
from model import MLP
from data_loader import load_mnist_data
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

def train_model(num_epochs: int = 10, lr: float = 0.01, batch_size: int = 64):
    """训练模型并保存损失曲线，支持学习率衰减"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_mnist_data(batch_size=batch_size)
    if train_loader is None:
        return None

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)  # 每5个epoch学习率乘以0.9

    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()  # 更新学习率
        print(f"Epoch {epoch+1} Complete, Average Loss: {epoch_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # 保存模型与损失数据
    torch.save(model.state_dict(), "mnist_model.pth")
    import numpy as np
    np.savetxt("train_losses.csv", train_losses, delimiter=",")
    return train_losses

if __name__ == "__main__":
    train_losses = train_model(num_epochs=15, lr=0.01, batch_size=128)
    if train_losses is not None:
        plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.savefig("loss_curve.png")
        plt.show()