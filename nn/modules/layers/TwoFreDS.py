import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU,
                 drop=0., bias=True):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = x.to(self.fc2.weight.dtype)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class RFBSpatial(nn.Module):
    def __init__(self, in_channel, norm='bn'):
        super(RFBSpatial, self).__init__()
        # 降维以减少计算量 (inter_channel = c // 4)
        inter_channel = in_channel // 4

        def make_norm(c):
            if norm == 'gn':
                g = 8 if c >= 8 else 1
                return nn.GroupNorm(g, c)
            return nn.BatchNorm2d(c)

        # 1x1 卷积
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1),
            make_norm(inter_channel),
            nn.ReLU(True)
        )

        #  3x3 标准卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1),
            make_norm(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1),
            make_norm(inter_channel),
            nn.ReLU(True)
        )

        #  3x3 膨胀卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1),
            make_norm(inter_channel),
            nn.ReLU(True),
            nn.Conv2d(inter_channel, inter_channel, 3, padding=3, dilation=3),
            make_norm(inter_channel),
            nn.ReLU(True)
        )

        # 将三个分支 concat 后还原通道数
        # 输入通道数是 inter_channel * 3
        self.conv_cat = nn.Conv2d(inter_channel * 3, in_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # 在通道维度拼接
        y = torch.cat((x0, x1, x2), 1)
        return self.conv_cat(y)

# TwoFreDS：进行了显式频域跨模态交互
class TwoFreDS(nn.Module):
    """
    - 在频域调制项 delta_weight 上加入“可学习的双向跨模态耦合项”
      使得 opt 的调制信号显式包含 sar 的调制信息（反之亦然）
动态的施加权重衰减，使模型只有在交互确实有效时才保留该权重。

      并且分成空域和频域混合分流
    """

    def __init__(self, dim, num_filters=4, group=16,
                 size_init=(256, 256), init_std=0.02, weight_limit=0.5,):
        super().__init__()
        # 将特征分为两半：一半走频域(Context)，一半走空域(Detail)
        self.freq_dim = dim // 2
        self.spat_dim = dim - self.freq_dim
        self.num_filters = num_filters
        self.group = group // 2  # group 数也减半
        self.channels_per_group = self.freq_dim // self.group
        self.weight_limit = weight_limit

        # 构建共享的“频域幅度基底库”
        # rfft2 的宽度维为 W//2+1（实数FFT的共轭对称导致）
        # bases 形状：[K, Cg, H0, W0_freq]
        freq_size_init = (size_init[0], size_init[1] // 2 + 1)
        self.freq_amplitude_bases = nn.Parameter(
            torch.randn(num_filters, self.channels_per_group, *freq_size_init, dtype=torch.float32) * init_std
        )
        trunc_normal_(self.freq_amplitude_bases, std=init_std)

        # 将第0个基底初始化为“温和高通先验”，帮助小目标高频细节更容易被增强
        self._init_high_pass_filter(filter_idx=0, size=freq_size_init, scale_factor=0.05, sigma=0.1)
        # 联合上下文归一化
        self.ctx_norm = nn.LayerNorm(self.freq_dim * 4)

        self.router = Mlp(dim=self.freq_dim * 4, out_features=2 * self.group * num_filters)


        # 动态交叉门控 (Cross-Modal Gated Interaction)
        # 而是根据上下文生成门控
        self.cross_modal_gate = nn.Sequential(
            nn.Linear(self.freq_dim * 4, self.group * 2),  # 为两个模态分别生成门控
            nn.Sigmoid()
        )
        # 每组一个交互强度系数，初始很小
        self.cross_scale_param = nn.Parameter(torch.full((1, self.group, 1, 1, 1), -2.0))
        # 每组一个频率调制强度，限制在较小正区间
        self.alpha_param = nn.Parameter(torch.zeros(1, self.group, 1, 1, 1))

        # 空域
        self.spatial_opt = RFBSpatial(self.spat_dim)
        self.spatial_sar = RFBSpatial(self.spat_dim)

        # 模态独立融合
        self.fusion_opt = nn.Conv2d(dim, dim, 1, bias=True)
        self.fusion_sar = nn.Conv2d(dim, dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 路由初始接近恒等
        nn.init.constant_(self.router.fc2.weight, 0.0)
        nn.init.constant_(self.router.fc2.bias, 0.0)

        nn.init.constant_(self.cross_modal_gate[0].weight, 0.0)
        nn.init.constant_(self.cross_modal_gate[0].bias, -1.0)

        # 融合层零初始化，初始近似恒等
        nn.init.constant_(self.fusion_opt.weight, 0)
        nn.init.constant_(self.fusion_opt.bias, 0)
        nn.init.constant_(self.fusion_sar.weight, 0)
        nn.init.constant_(self.fusion_sar.bias, 0)

        # 空域支路初始输出接近 0
        nn.init.constant_(self.spatial_opt.conv_cat.weight, 0)
        if self.spatial_opt.conv_cat.bias is not None:
            nn.init.constant_(self.spatial_opt.conv_cat.bias, 0)

        nn.init.constant_(self.spatial_sar.conv_cat.weight, 0)
        if self.spatial_sar.conv_cat.bias is not None:
            nn.init.constant_(self.spatial_sar.conv_cat.bias, 0)

    def _init_high_pass_filter(self, filter_idx, size, scale_factor=0.05, sigma=0.1):
        """
        构造一个“温和的高通”权重图，赋给某个基底（默认第0个）。
        离(0,0)直流分量越远，权重越大，更强调高频细节。
        """
        H, W_freq = size
        y = torch.arange(H, dtype=torch.float32)
        x = torch.arange(W_freq, dtype=torch.float32)
        # indexing='ij' 可避免新版 torch 的 meshgrid 默认行为变化
        grid_y, grid_x = torch.meshgrid(y, x)

        # H 维频率具有周期性：取 min(y, H-y) 更符合真实频率距离
        dist_y = torch.min(grid_y, H - grid_y)
        dist_x = grid_x  # rfft 只有正半轴
        distance = torch.sqrt(dist_y ** 2 + dist_x ** 2)

        # 归一化到[0,1]
        normalized_dist = distance / (distance.max() + 1e-6)

        # 高通曲线：中心低，边缘高（sigma控制上升快慢）
        weight_map = 1.0 - torch.exp(-(normalized_dist.pow(2)) / (2 * sigma ** 2))
        weight_map = weight_map * scale_factor  # 缩放强度，避免初期震荡

        with torch.no_grad():
            # 扩展到 [Cg, H, W_freq]
            self.freq_amplitude_bases.data[filter_idx] = weight_map.unsqueeze(0).expand(
                self.channels_per_group, -1, -1
            )

    def _get_context(self, x):
        """
        提取全局上下文特征：mean + std
        - mean：全局分布信息（偏低频/整体）
        - std：纹理/细节强度（对小目标、噪声模式敏感）
        输出形状：[B, 2C]
        """
        ctx_mean = x.mean(dim=(2, 3))
        ctx_std = x.std(dim=(2, 3))
        return torch.cat([ctx_mean, ctx_std], dim=1)

    def _resize_bases_if_needed(self, H_freq, W_freq):
        bases = self.freq_amplitude_bases.float()
        if bases.shape[-2:] != (H_freq, W_freq):
            bases = F.interpolate(
                bases, size=(H_freq, W_freq), mode='bicubic', align_corners=True
            )
        return bases

    def forward_freq(self, x_opt, x_sar):
        """
        输入：
          x_opt, x_sar: [B, C, H, W]
        输出：
          enhanced_opt, enhanced_sar: [B, C, H, W]
        """
        B, C, H, W = x_opt.shape
        input_dtype = x_opt.dtype

        x_opt_f = x_opt.float()
        x_sar_f = x_sar.float()
        #  FFT：空域 -> 频域（复数）
        # 输出形状 [B, C, H, W//2+1]
        fft_opt = torch.fft.rfft2(x_opt_f, dim=(2, 3), norm='ortho')
        fft_sar = torch.fft.rfft2(x_sar_f, dim=(2, 3), norm='ortho')
        H_freq, W_freq = fft_opt.shape[-2:]

        # 联合全局感知：用 opt/sar 的 mean+std 拼接作为 router 输入
        ctx_opt = self._get_context(x_opt_f)  # [B, 2C]
        ctx_sar = self._get_context(x_sar_f)  # [B, 2C]
        fuse_ctx = torch.cat([ctx_opt, ctx_sar], dim=1)  # [B, 4C]
        fuse_ctx = self.ctx_norm(fuse_ctx.to(self.ctx_norm.weight.dtype))

        # routes = F.leaky_relu(self.router(fuse_ctx))
        # tanh 控制范围在(-1,1)，便于正负叠加基底
        linear_dtype = self.router.fc1.weight.dtype
        fuse_ctx_linear = fuse_ctx.to(linear_dtype)

        routes = self.router(fuse_ctx_linear).view(B, 2, self.group, self.num_filters)
        routes = torch.tanh(routes).float()

        gates = self.cross_modal_gate(fuse_ctx_linear).view(B, 2, self.group, 1, 1, 1).float()
        gate_opt = gates[:, 0]
        gate_sar = gates[:, 1]

        bases = self._resize_bases_if_needed(H_freq, W_freq)

        #  由 routes 与 bases 线性组合得到“频域调制项 delta_weight”
        # 输出先是分组形式 [B, G, Cg, Hf, Wf]
        # 基础调制项 delta (B, G, Cg, Hf, Wf)
        # [B, G, Cg, Hf, Wf]
        d_opt = torch.einsum('bgk,kchw->bgchw', routes[:, 0], bases)
        d_sar = torch.einsum('bgk,kchw->bgchw', routes[:, 1], bases)

        # 小幅度可学习跨模态交互
        cross_scale = 0.25 * torch.sigmoid(self.cross_scale_param).float()
        interleaved_opt = d_opt + cross_scale * gate_opt * d_sar
        interleaved_sar = d_sar + cross_scale * gate_sar * d_opt

        # 有界频域调制强度
        alpha = 0.1 * torch.sigmoid(self.alpha_param).float()

        # 更稳的权重参数化：围绕 1 波动，避免 exp 过激
        weight_opt = 1.0 + self.weight_limit * torch.tanh(alpha * interleaved_opt)
        weight_sar = 1.0 + self.weight_limit * torch.tanh(alpha * interleaved_sar)

        weight_opt = weight_opt.reshape(B, C, H_freq, W_freq)
        weight_sar = weight_sar.reshape(B, C, H_freq, W_freq)

        # 实数权重乘复数频谱：只改幅度，不改相位
        fft_opt_enhanced = fft_opt * weight_opt
        fft_sar_enhanced = fft_sar * weight_sar

        out_opt = torch.fft.irfft2(fft_opt_enhanced, s=(H, W), dim=(2, 3), norm='ortho')
        out_sar = torch.fft.irfft2(fft_sar_enhanced, s=(H, W), dim=(2, 3), norm='ortho')

        return out_opt.to(input_dtype), out_sar.to(input_dtype)

    def forward(self, x,x_sar=None):
        if isinstance(x, list):
            x_opt, x_sar = x
        else:
            x_opt = x
            if x_sar is None:
                raise ValueError("TwoFreDS requires two input modalities.")
        c = x_opt.shape[1]  # 获取输入的光学特征图实际通道数
        f_dim = c // 2  # 取一半给频域
        s_dim = c - f_dim  # 剩下的一半给空域
        # x_opt_freq: [B, C/2, H, W] 去做全局交互
        # x_opt_spat: [B, C/2, H, W]  保留小目标细节
        x_opt_freq, x_opt_spat = torch.split(x_opt, [f_dim, s_dim], dim=1)
        x_sar_freq, x_sar_spat = torch.split(x_sar, [f_dim, s_dim], dim=1)

        # 频域增强 (处理全局、对齐、大目标)
        out_opt_freq, out_sar_freq = self.forward_freq(x_opt_freq, x_sar_freq)

        #  空域增强 (处理小目标、纹理)
        # 加上残差 x_opt_spat 确保原始高频细节不丢失
        out_opt_spat = self.spatial_opt(x_opt_spat) + x_opt_spat
        out_sar_spat = self.spatial_sar(x_sar_spat) + x_sar_spat

        # Concat
        out_opt = torch.cat([out_opt_freq, out_opt_spat], dim=1)
        out_sar = torch.cat([out_sar_freq, out_sar_spat], dim=1)

        # 因为刚才通道被分开处理了，最后用 1x1
        out_opt = x_opt + self.fusion_opt(out_opt)
        out_sar = x_sar + self.fusion_sar(out_sar)

        return torch.cat([out_opt, out_sar], dim=1)

if __name__ == "__main__":
    x_opt = torch.randn(2, 64, 128, 128)
    x_sar = torch.randn(2, 64, 128, 128)

    # 推荐：cross_mix_mode="group"，初始交互强度为 0，训练更稳
    model = TwoFreDS(dim=64, num_filters=2, group=16, size_init=(128, 128))

    out_opt, out_sar = model(x_opt, x_sar)
    print("Input:", x_opt.shape, "Output:", out_opt.shape)

    loss = out_opt.sum() + out_sar.sum()
    loss.backward()
    print("Backward OK.")
