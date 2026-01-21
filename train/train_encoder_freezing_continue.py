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
        start_epoch: int = 1  # [修改1] 新增参数：起始 Epoch
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

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Start Epoch:     {start_epoch}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # [修改2] 智能日志模式：如果是第1轮则覆盖('w')，否则追加('a')
    if start_epoch > 1:
        logging.info(f"Resuming logging to training_log.txt from epoch {start_epoch}...")
        log_file = open("training_log.txt", "a", encoding="utf-8")
        # 追加模式不需要再写表头
    else:
        log_file = open("training_log.txt", "w", encoding="utf-8")
        log_file.write("Epoch,Train_Loss,Val_Score\n") 

    # 4. Set up the optimizer... (保持不变)
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    patience = 30
    patience_counter = 0
    best_val_score = 0.0

    # 5. Begin training
    # [修改3] 循环从 start_epoch 开始
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # ... (内部训练逻辑保持完全不变，省略以节省空间) ...
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
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # ... (验证逻辑保持不变) ...
        avg_train_loss = epoch_loss / len(train_loader)
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info(f'Validation Dice score: {val_score}')

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f'Early Stopping counter: {patience_counter} out of {patience}')
            if patience_counter >= patience:
                logging.info('Early stopping triggered! Training stopped.')
                break 
        
        log_file.write(f"{epoch},{avg_train_loss:.4f},{val_score:.4f}\n")
        log_file.flush()
        
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            # 这里文件名会自动使用当前的 epoch (如 73, 74...)
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    log_file.close()


if __name__ == '__main__':
    # 引入 re 模块用于正则提取数字
    import re 
    
    args = get_args()
    set_seed(42)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    # 冻结 Encoder 层代码... (保持不变)
    logging.info("Freezing Encoder parameters (inc, down1-4)...")
    for name, param in model.named_parameters():
        if 'inc' in name or 'down' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # [修改4] 计算 start_epoch
    start_epoch = 1
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
        
        # 尝试从文件名中提取 epoch 数。例如 "checkpoints/checkpoint_epoch72.pth" -> 72
        match = re.search(r'epoch(\d+)', args.load)
        if match:
            loaded_epoch = int(match.group(1))
            start_epoch = loaded_epoch + 1  # 如果读取的是 72，那么从 73 开始
            logging.info(f'Detected checkpoint from epoch {loaded_epoch}. Resuming start from epoch {start_epoch}.')
        else:
            logging.info(f'Could not extract epoch number from filename. Defaulting start_epoch to 1.')

    model.to(device=device)
    
    # 训练逻辑
    try:
        train_model(
            model=model,
            epochs=args.epochs, # 注意：这里的 epochs 应该是你想训练到的 *总目标轮数* (比如100)
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            start_epoch=start_epoch # [修改5] 传入起始轮数
        )
    except torch.cuda.OutOfMemoryError:
        # 同样的修改应用到这里
        logging.error('Detected OutOfMemoryError! Enabling checkpointing...')
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
            amp=args.amp,
            start_epoch=start_epoch # [修改5] 传入起始轮数
        )