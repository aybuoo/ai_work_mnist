# visualize.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from data_loader import load_mnist_data

def visualize_data_samples():
    """可视化MNIST数据集中的0-9数字示例"""
    _, test_loader = load_mnist_data(batch_size=10)  # 取10个不同类别样本
    if test_loader is None:
        return

    images, labels = next(iter(test_loader))  # 获取第一批次数据（假设包含0-9）
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i][0], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.suptitle("MNIST Dataset Sample Visualization")
    plt.savefig("data_samples.png")
    plt.show()

def plot_loss_curve(loss_file: str = "train_losses.csv"):
    """绘制训练损失曲线"""
    train_losses = np.loadtxt(loss_file, delimiter=",")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, "bo-", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss vs. Epochs")
    plt.grid(alpha=0.3)
    plt.savefig("loss_curve.png")
    plt.show()

def plot_confusion_matrix(conf_matrix_file: str = "confusion_matrix.csv"):
    """绘制混淆矩阵热力图"""
    conf_matrix = np.loadtxt(conf_matrix_file, delimiter=",", dtype=int)
    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay(conf_matrix, display_labels=list(range(10))).plot(cmap="Blues")
    plt.title("Confusion Matrix (First 100 Test Samples)")
    plt.savefig("confusion_matrix.png")
    plt.show()

def visualize_error_cases(error_file: str = "error_samples.npy"):
    """可视化错误分类案例"""
    error_samples = np.load(error_file, allow_pickle=True)
    num_errors = len(error_samples)
    if num_errors == 0:
        print("No error cases found in the first 100 test samples.")
        return

    plt.figure(figsize=(12, num_errors * 2))
    for i, (img, true_lab, pred_lab) in enumerate(error_samples):
        plt.subplot(num_errors, 1, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {true_lab}, Pred: {pred_lab}", color="red" if true_lab != pred_lab else "green")
        plt.axis("off")
    plt.suptitle("Misclassified Samples Analysis")
    plt.savefig("error_cases.png")
    plt.show()

if __name__ == "__main__":
    visualize_data_samples()
    plot_loss_curve()
    plot_confusion_matrix()
    visualize_error_cases()