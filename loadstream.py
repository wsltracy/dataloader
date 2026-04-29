from torch.utils.data import DataLoader
from streaming import StreamingDataset
from torchvision import transforms
import torch

data_dir = './datas/stream/1'

# 简单的 collate 函数
def collate_fn(batch):
    images = torch.stack([transforms.ToTensor()(item['image']) for item in batch])
    labels = torch.tensor([item['class'] for item in batch])
    return images, labels  # 直接返回 tuple 而不是 dict

dataset = StreamingDataset(
    local=data_dir,
    shuffle=True,
    batch_size=32
)
print(f"数据集总样本数: {len(dataset)}")
print(f"样本字段: {dataset[0].keys()}")
print(f"图片类型: {type(dataset[0]['image'])}")
print(f"图片尺寸: {dataset[0]['image'].size}")
print(f"类别标签: {dataset[0]['class']}")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,
    collate_fn=collate_fn
)

# 使用时
for images, labels in dataloader:
    print(f"Images: {images.shape}, Labels: {labels.shape}")
    # images: [32, 3, 32, 32], labels: [32]
    break