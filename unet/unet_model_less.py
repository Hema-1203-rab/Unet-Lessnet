from .unet_parts import *

class LessNet(nn.Module):
    def __init__(self, n_channels, n_classes, C=8, bilinear=False):
        super(LessNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 论文中的超参数 C 控制通道宽度 [cite: 300, 336]
        # 每层通道数参考论文：4C, 3C, 2C, C [cite: 242, 244, 247, 249]
        
        # 1. 第一块：输入是 1/8 分辨率的池化特征 [cite: 241]
        # 假设输入 1 通道，经过 3 种池化拼接后为 3 通道 (如果是配准则是 2*3=6) [cite: 195, 198]
        pool_ch = n_channels * 3 
        
        self.block1 = nn.Sequential(
            nn.Conv2d(pool_ch, 4 * C, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 2. 第二块：上采样并拼接 1/4 池化特征 [cite: 244]
        self.up1 = Up(in_channels=(4 * C) + pool_ch, out_channels=3 * C, bilinear=bilinear)
        
        # 3. 第三块：上采样并拼接 1/2 池化特征 [cite: 246]
        self.up2 = Up(in_channels=(3 * C) + pool_ch, out_channels=2 * C, bilinear=bilinear)
        
        # 4. 第四块：上采样并拼接原始图像 [cite: 249]
        self.up3 = Up(in_channels=(2 * C) + n_channels, out_channels=C, bilinear=bilinear)
        
        # 输出层
        self.outc = OutConv(C, n_classes)

    def forward(self, inputs): # 1. 将参数名改为 inputs，避免混淆
        # 生成多尺度池化特征
        # 使用 inputs 而不是 x
        feat_s2 = PoolingFeatures(kernel_size=2, stride=2)(inputs)
        feat_s4 = PoolingFeatures(kernel_size=4, stride=4)(inputs)
        feat_s8 = PoolingFeatures(kernel_size=8, stride=8)(inputs)

        # Decoder 路径
        # 2. 从这里开始，使用 x 来存储中间结果
        x = self.block1(feat_s8)            # 1/8 分辨率处理
        x = self.up1(x, feat_s4)            # 升至 1/4 并拼接
        x = self.up2(x, feat_s2)            # 升至 1/2 并拼接
        
        # 3. 关键修改：第二个参数传回原始的 inputs (512x512)
        x = self.up3(x, inputs)             # 升至全分辨率并拼接原图
        
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)