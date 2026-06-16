import torch
import torch.nn as nn


class Fusion_Module(nn.Module):
    """
    特征聚合 (Concat + 1x1 Conv)
    双路权重生成 (Dual-Path Weight Generation):
       - Global Path: 捕捉环境上下文 (Avg -> MLP)
       - Local Path: 保留微小目标 (1x1 Conv)
    动态互补融合 (Dynamic Complementary Fusion)
    允许两模态在同一位置/通道同时贡献，从而与前面 TwoFreDS 的“互补增强”保持一致
    细节重绘 (Detail Refinement with Residual)
    """

    def __init__(self, dim, reduction=16):
        super().__init__()

        # 将 OPT 和 SAR 拼接后融合，准备提取共性特征
        self.pre_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # 全局部分
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享 MLP (Shared MLP) 用于处理全局特征
        self.global_mlp = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim * 2, 1)
        )

        # 不进行下采样，保留空间位置信息，局部部分
        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            # 引入一个3x3的深度卷积
            nn.Conv2d(dim // reduction, dim // reduction, 3, padding=1,
                  groups=dim // reduction, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim * 2, 1)

        )

        # 激活函数，生成 0~1 的权重
        self.sigmoid = nn.Sigmoid()

        # 使用 5x5 深度卷积Depthwise Conv
        self.refine_dw = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
            # nn.Conv2d(dim, dim, 1, bias=False),
            # nn.BatchNorm2d(dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN 层标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-Init 最后一个 BN
        # 让 refine_dw 一开始输出为 0，变成恒等映射
        if isinstance(self.refine_dw[-2], nn.BatchNorm2d):
            nn.init.constant_(self.refine_dw[-2].weight, 0)
            nn.init.constant_(self.refine_dw[-2].bias, 0)

    def forward(self, x):
        if isinstance(x, list):
            x_opt, x_sar = x
        else:
            # 防止报错，如果只传了一个输入，就自己复制一份
            x_opt = x_sar = x
        # 拼接与预处理
        x_cat = torch.cat([x_opt, x_sar], dim=1)
        x_fused = self.pre_conv(x_cat)

        # 计算全局路径特征 GAP GMP
        avg_out = self.global_avg_pool(x_fused)
        max_out = self.global_max_pool(x_fused)
        global_pool_feat = torch.cat([avg_out, max_out], dim=1)
        global_feat = self.global_mlp(global_pool_feat)

        #  计算局部路径特征 (1x1)
        local_feat = self.local_path(x_fused)
        # local_feat shape: [B, C, H, W]

        # 这一步网络会同时考虑"整体环境"和"局部亮点"
        w = self.sigmoid(global_feat + local_feat)
        g_opt, g_sar = torch.chunk(w, chunks=2, dim=1)

        fused = g_opt * x_opt + g_sar * x_sar
        # 加上残差非常重要，防止梯度消失，且让卷积层专注于"修补"工作
        out = fused + self.refine_dw(fused)

        return out
