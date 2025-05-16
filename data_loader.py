# data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def load_mnist_data(batch_size: int = 64, data_dir: str = "./mnist"):
    """
    加载并预处理MNIST数据集，支持自动下载和手动下载
    :param batch_size: 批量大小
    :param data_dir: 数据集保存路径
    :return: 训练数据加载器, 测试数据加载器
    """
    # 数据预处理：转为张量、归一化、数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # 官方统计值
        transforms.RandomAffine(15, translate=(0.1, 0.1))  # 随机旋转±15°，平移10%
    ])

    # 检查数据集文件是否存在
    required_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
        print("MNIST数据集未找到，请确保以下文件存在于目录中：")
        print(f"{os.path.abspath(data_dir)}/")
        for f in required_files:
            print(f"- {f}")
        print("下载地址：http://yann.lecun.com/exdb/mnist/")
        return None, None

    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 加载数据集，允许自动下载
    try:
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请检查数据集是否已正确放置到以下目录：")
        print(f"{os.path.abspath(data_dir)}/raw")
        return None, None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
