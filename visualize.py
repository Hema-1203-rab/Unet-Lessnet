import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path

# 导入你项目中的 U-Net 定义
from unet import UNet

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'Random seed set to {seed}')

def plot_feature_maps(features, layer_name, seed, model_name):
    """
    封装绘图逻辑：展示前12个通道
    """
    features = features.squeeze(0).cpu()  # [Channels, H, W]
    num_channels = features.shape[0]
    display_count = min(12, num_channels)
    
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle(f'Layer: {layer_name} | Seed: {seed} | Weights: {model_name}', fontsize=16)
    
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < display_count:
            # magma 能够很好地展示特征图的激活强度
            ax.imshow(features[i].detach().numpy(), cmap='magma') 
            ax.set_title(f'Channel {i}')
        ax.axis('off')
    
    plt.tight_layout()
    output_name = f"viz_{layer_name}_seed{seed}.png"
    plt.savefig(output_name)
    plt.show()
    print(f"Saved: {output_name}")

def visualize_all_encoder_layers(model_path, image_path, seed=114, device='cuda'):
    # 1. 环境准备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # 2. 实例化并加载模型
    model = UNet(n_channels=3, n_classes=2, bilinear=False) 
    
    # 加载权重
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 图像预处理
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 4. 前向传播并逐层提取
    with torch.no_grad():
        x1 = model.inc(img_tensor)
        d1 = model.down1(x1)  # Down 1
        d2 = model.down2(d1)  # Down 2
        d3 = model.down3(d2)  # Down 3
        d4 = model.down4(d3)  # Down 4

    # 5. 分别可视化四张图
    layers_to_viz = {
        "Down1": d1,
        "Down2": d2,
        "Down3": d3,
        "Down4": d4
    }

    model_name = Path(model_path).name
    for name, feat in layers_to_viz.items():
        plot_feature_maps(feat, name, seed, model_name)

if __name__ == '__main__':
    CHECKPOINT_FILE = './checkpoints/Unet_encoder_freezing_seed42/checkpoint_epoch44.pth' 
    TEST_IMAGE = './data/imgs/21_training.tif'
    SEED_TO_TEST = 42
    
    visualize_all_encoder_layers(CHECKPOINT_FILE, TEST_IMAGE, seed=SEED_TO_TEST)