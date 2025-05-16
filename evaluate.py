# evaluate.py
import torch
from model import MLP
from data_loader import load_mnist_data
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(load_model_path: str = "mnist_model.pth", top_n: int = 100):
    """评估模型并生成混淆矩阵和错误案例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_mnist_data()
    if test_loader is None:
        return None, None, None

    model = MLP().to(device)
    model.load_state_dict(torch.load(load_model_path))
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    error_samples = []  # 存储错误样本（图像、真实标签、预测标签）

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 记录前top_n个样本的错误案例
            if i < top_n // test_loader.batch_size + 1:
                for img, true_lab, pred_lab in zip(images, labels, preds):
                    if true_lab != pred_lab:
                        error_samples.append((img.cpu().numpy().squeeze(), true_lab.item(), pred_lab.item()))

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # 计算前100个样本的混淆矩阵
    test_labels_first_100 = all_labels[:top_n]
    pred_labels_first_100 = all_preds[:top_n]
    conf_matrix = confusion_matrix(test_labels_first_100, pred_labels_first_100)

    # 保存结果
    np.savetxt("confusion_matrix.csv", conf_matrix, fmt="%d", delimiter=",")
    np.save("error_samples.npy", np.array(error_samples, dtype=object))  # 保存错误样本数据

    return accuracy, conf_matrix, error_samples

if __name__ == "__main__":
    evaluate_model()