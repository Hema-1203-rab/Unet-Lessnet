import argparse
import logging
import os
import random
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt  # [新增] 引入绘图库

# [移除] import wandb 及其相关依赖

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info(f'Random seed set to {seed}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # [修改] 移除 WandB 初始化，改为本地日志初始化
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # [新增] 初始化本地记录字典和文件
    log_file = open("training_log.txt", "w", encoding="utf-8")
    log_file.write("Epoch,Train_Loss,Val_Score\n") # 写入表头

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    patience = 1000          # 容忍多少轮不提升
    patience_counter = 0   # 当前计数
    best_val_score = 0.0   # 历史最高分

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                # [移除] experiment.log 相关代码
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # [修改] 验证环节移至 Epoch 循环末尾，确保每个 Epoch 记录一次
        # 计算该 Epoch 的平均训练 Loss
        avg_train_loss = epoch_loss / len(train_loader)
        
        '''
        # 1. 随便拿一个 batch 的数据做测试
        model.eval()
        x_val = next(iter(val_loader))['image'].to(device)
        with torch.no_grad():
            output_val = model(x_val)

        # 2. 检查输出情况
        if model.n_classes > 1:
            probs = F.softmax(output_val, dim=1)
            pred = torch.argmax(probs, dim=1)
        else:
            probs = torch.sigmoid(output_val)
            pred = (probs > 0.5).float()

        # 3. 打印关键信息
        print(f"\n[Debug] Epoch {epoch}:")
        print(f"  - Unique prediction values: {torch.unique(pred)}")
        print(f"  - Max probability: {probs.max().item():.4f}")
        '''

        # 执行验证
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)

        logging.info(f'Validation Dice score: {val_score}')

        # [修改] 保存最佳模型与早停逻辑合并
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            
            # --- 核心修改开始 ---
            # 只有当当前分数 > 历史最高分时，才执行保存
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                
                # 将文件名固定为 checkpoint_best.pth，这样会自动覆盖旧的最佳模型
                save_path = dir_checkpoint / 'checkpoint_best.pth'
                torch.save(state_dict, str(save_path))
                
                logging.info(f'New best model saved! (Epoch {epoch}, Dice: {val_score:.4f})')
            # --- 核心修改结束 ---
            
        else:
            patience_counter += 1
            logging.info(f'Early Stopping counter: {patience_counter} out of {patience}')
            if patience_counter >= patience:
                logging.info('Early stopping triggered! Training stopped.')
                break  # 跳出 epoch 循环
        
        # [新增] 写入本地日志文件
        log_file.write(f"{epoch},{avg_train_loss:.4f},{val_score:.4f}\n")
        log_file.flush() # 强制刷新缓存写入硬盘

    # [新增] 训练结束后关闭文件并画图
    log_file.close()


if __name__ == '__main__':
    args = get_args()
    set_seed(114)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    # [新增] 2. 冻结 Encoder 层的参数
    logging.info("Freezing Encoder parameters (inc, down1-4)...")
    for name, param in model.named_parameters():
        # 判断参数名是否包含 'inc' 或 'down'，如果包含则冻结
        if 'inc' in name or 'down' in name:
            param.requires_grad = False
        else:
            # 确保 Decoder 部分是可训练的 (默认就是 True，这里是为了显式确认)
            param.requires_grad = True

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )