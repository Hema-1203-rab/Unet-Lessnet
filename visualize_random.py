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

def plot_random_channels(features, layer_name, seed, model_name, num_to_sample=24):
    """
    每一层随机抽取24个通道并展示
    """
    features = features.squeeze(0).cpu()  # [Channels, H, W]
    total_channels = features.shape[0]
    
    # 随机选取通道索引
    # 如果总通道数小于24（比如输入层或极浅层），则取全部
    sample_size = min(num_to_sample, total_channels)
    # 使用固定的 random.sample 配合 seed 可以保证每次运行结果一致
    indices = sorted(random.sample(range(total_channels), sample_size))
    
    # 设置 4行 6列
    rows, cols = 4, 6
    fig, axes = plt.subplots(rows, cols, figsize=(20, 14))
    fig.suptitle(f'Layer: {layer_name} | Random {sample_size} Channels | Seed: {seed}', fontsize=20)
    
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < sample_size:
            channel_idx = indices[i]
            # 这里的特征图进行简单的归一化，让明暗对比更显著
            f_img = features[channel_idx].detach().numpy()
            f_min, f_max = f_img.min(), f_img.max()
            f_img = (f_img - f_min) / (f_max - f_min + 1e-8)
            
            ax.imshow(f_img, cmap='magma')
            ax.set_title(f'Ch {channel_idx}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为总标题留出空间
    output_name = f"viz_{layer_name}_random24_seed{seed}.png"
    plt.savefig(output_name, dpi=150) # 提高分辨率
    plt.show()
    print(f"Saved: {output_name}")

def visualize_encoder_layers_random(model_path, image_path, seed=114, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # 1. 加载模型
    model = UNet(n_channels=3, n_classes=2, bilinear=False) 
    state_dict = torch.load(model_path, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. 预处理
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # 3. 推理提取
    with torch.no_grad():
        x1 = model.inc(img_tensor)
        d1 = model.down1(x1)
        d2 = model.down2(d1)
        d3 = model.down3(d2)
        d4 = model.down4(d3)

    # 4. 循环可视化
    layers = {"Down1": d1, "Down2": d2, "Down3": d3, "Down4": d4}
    model_name = Path(model_path).name
    
    for name, feat in layers.items():
        plot_random_channels(feat, name, seed, model_name)

if __name__ == '__main__':
    # 替换为你自己的路径
    CHECKPOINT_FILE = './checkpoints/Unet_seed2026/checkpoint_epoch44.pth' 
    TEST_IMAGE = './data/imgs/21_training.tif'
    SEED_TO_TEST = 2026
    
    visualize_encoder_layers_random(CHECKPOINT_FILE, TEST_IMAGE, seed=SEED_TO_TEST)