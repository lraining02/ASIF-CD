# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import os
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

# 模型预测的是一个“分布”（16个概率），而不是一个具体的数
class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max # 默认16，即把距离切分成0~15个整数格子

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        # 向下取整
        tl = target.long()  # target left
        # 向上取整
        tr = tl + 1  # target right
        # 向下取整的权重
        wl = tr - target  # weight left
        # 向上取整的权重
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

# 回归损失 它同时计算 IoU Loss 和 DFL Loss，并把它们加权组合
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss.
        参数解释:
        - pred_dist: 预测的 DFL 分布 (用于算 DFL Loss)
        - pred_bboxes: 解码后的预测框 (xyxy, 用于算 IoU Loss)
        - anchor_points: 网格中心点 (用于把 GT 框转成距离)
        - target_bboxes: GT 框
        - target_scores: TAL 分配后的软标签分数 (用于加权)
        - fg_mask: 正样本掩码
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 计算预测框和 GT 框的 CIoU
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # pred_dist[fg_mask]: 只取正样本的预测分布
            # target_ltrb[fg_mask]: 只取正样本的目标距离
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        # # 获取模型的最后一层 (Detect Head)
        m = model.model[-1]  # Detect() module
        # 定义分类损失函数 (BCE With Logits)，用于多标签分类
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides 下采样倍率 (8, 16, 32)
        # self.stride = torch.tensor([4., 8., 16.], device=device)
        self.nc = m.nc  # number of classes
        # 计算每个 anchor 的输出通道数 = 类别数 + 4 * 16 (DFL分布)
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        # 这是 YOLOv8 的核心，决定哪些网格负责预测物体
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        # 初始化边界框损失 (IoU Loss + DFL Loss)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # 初始化 DFL 积分用的投影张量 [0, 1, ..., 15]
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # 初始化 3 个 Loss 的容器：[box_loss, cls_loss, dfl_loss]
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        # feats 里的每个特征图形状是 [B, C, H, W]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        # 变成 [Batch, Anchors, Channels]，方便后续计算
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # 获取数据类型和 batch 大小
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # 计算特征图对应的输入图片尺寸 (用于后续坐标还原)
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # anchor_points: 所有网格点的中心坐标 (x, y) 实际上是anchorfree的，我们只有每个网格的中心坐标
        # stride_tensor: 每个网格点对应的下采样倍率 (8, 16, 32)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # 把 targets 对齐到当前 batch，并把坐标归一化转为绝对像素坐标
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes 把预测的变成框，因为有预测的距离
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss # 使用 BCE 计算预测分数和目标分数 (Tal 分配出来的) 的差距
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class MultiLoss(v8DetectionLoss):
    def __init__(self, model, use_distillation=True):
        super().__init__(model)
        self.use_distillation = use_distillation
        self.training = True
    def __call__(self, preds, batch):
        """
        preds: 接收来自 MultiDetect 的字典
        batch: 标签数据
        """
        if isinstance(preds, dict):
            # 分别提取三路预测值
            feats_fus = preds["fus"]
            batch_size = feats_fus[0].shape[0]  # 获取 Batch 大小
            device = feats_fus[0].device

            # 计算融合分支的主 Loss
            # 调用父类的逻辑，它会处理正样本分配 (TAL)
            loss_fus, loss_items, intermediate_data = self.get_detailed_loss(feats_fus, batch)

            # 计算 Optical 和 SAR 的辅助 Loss
            # 直接调用 super().__call__ (即 v8DetectionLoss)，将其作为辅助任务训练
            loss_opt, _ = super().__call__(preds["opt"], batch)
            # loss_opt = raw_loss_opt / batch_size
            loss_sar, _ = super().__call__(preds["sar"], batch)
            # loss_sar = raw_loss_sar / batch_size
            loss_distill = torch.tensor(0.0).to(device)
            # 是否开启蒸馏
            if self.use_distillation:
                # 提取中间变量
                fg_mask = intermediate_data['fg_mask']  # [B, Anchors] 前景掩码
                target_scores = intermediate_data['target_scores']  # [B, Anchors, NC] # 每个点的分类标签
                target_bboxes = intermediate_data['target_bboxes']  # [B, Anchors, 4]  # 每个格点的回归目标

                # 只选取每个目标中最匹配的一个格点进行学习，而不是所有正样本
                top1_mask = torch.zeros_like(fg_mask, dtype=torch.bool,device=device)
                anchor_scores = target_scores.max(-1)[0]  # TAL 计算出的融合分 (s^alpha * IoU^beta)

                with torch.no_grad():
                    for i in range(batch_size):
                        img_fg_mask = fg_mask[i]
                        if not img_fg_mask.any():
                            continue

                        # 找到当前图片所有正样本索引及对应的 GT 框
                        fg_idx = torch.where(img_fg_mask)[0]
                        img_gt_boxes = target_bboxes[i, fg_idx]

                        # 利用 GT 框的唯一性区分不同实例 (同一目标的格点具有相同的 target_bboxes)
                        unique_boxes, inverse_indices = torch.unique(
                            img_gt_boxes, dim=0, return_inverse=True
                        )

                        for instance_idx in range(len(unique_boxes)):
                            # 提取属于该实例的所有格点
                            instance_indices = fg_idx[inverse_indices == instance_idx]
                            # 在该实例内部，选出对齐得分最高的格点索引
                            best_idx = instance_indices[torch.argmax(anchor_scores[i, instance_indices])]
                            top1_mask[i, best_idx] = True

                # 只对正样本点进行比拼和蒸馏
                if top1_mask.any():
                    # 提取正样本点的数据  获取三路预测的原始 Logits (分类) 和分布 (回归)
                    dist_fus, score_fus = self.slice_preds(preds["fus"])
                    dist_opt, score_opt = self.slice_preds(preds["opt"])
                    dist_sar, score_sar = self.slice_preds(preds["sar"])

                    # 超参 
                    teacher_threshold = 0.9  # 教师置信度阈值 tau
                    T = 2.0
                    distill_weight = 0.05  # lambda

                    with torch.no_grad():
                        # 基于表现比拼
                        qual_fus = (score_fus.sigmoid() * target_scores).sum(-1)
                        qual_opt = (score_opt.sigmoid() * target_scores).sum(-1)
                        qual_sar = (score_sar.sigmoid() * target_scores).sum(-1)
                        # teacher_mask: 1 代表 Optical 更好, 0 代表 SAR 更好
                        qual_fus_top1 = qual_fus[top1_mask]
                        qual_opt_top1 = qual_opt[top1_mask]
                        qual_sar_top1 = qual_sar[top1_mask]

                        max_qual_teacher, teacher_idx = torch.max(torch.stack([qual_opt_top1, qual_sar_top1]), dim=0) # [B, Anchors, 1]
                        valid_distill_mask = (max_qual_teacher > teacher_threshold) & (max_qual_teacher > qual_fus_top1)

                    if valid_distill_mask.any():
                        # 只取那些通过阈值筛选的样本进行计算
                        # teacher_mask: 1 代表 Optical 更优，0 代表 SAR 更优
                        teacher_mask_top1 = (
                                    qual_opt_top1[valid_distill_mask] > qual_sar_top1[valid_distill_mask]
                        ).float().view(-1,
                                                                                                                        1)

                        logits_fus = score_fus[top1_mask][valid_distill_mask]
                        logits_opt = score_opt[top1_mask][valid_distill_mask]
                        logits_sar = score_sar[top1_mask][valid_distill_mask]

                        # 动态选择老师的分类 Logits
                        logits_teacher = teacher_mask_top1 * logits_opt.detach()\
                                         + (1.0 - teacher_mask_top1) * logits_sar.detach()

                        # 使用 KL 散度对齐概率分布 分类蒸馏
                        loss_distill_cls = F.kl_div(
                            F.log_softmax(logits_fus / T, dim=-1),
                            F.softmax(logits_teacher.detach() / T, dim=-1),
                            reduction='batchmean'
                        ) * (T ** 2)

                        d_fus = dist_fus[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)
                        d_opt = dist_opt[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)
                        d_sar = dist_sar[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)

                        teacher_mask_reg = teacher_mask_top1.view(-1, 1, 1)

                        # 动态选择老师的回归分布
                        d_teacher = teacher_mask_reg * d_opt.detach() \
                                    + (1.0 - teacher_mask_reg) * d_sar.detach()
                        # 回归蒸馏
                        loss_distill_reg = F.kl_div(
                            F.log_softmax(d_fus.view(-1, self.reg_max), dim=-1),
                            F.softmax(d_teacher.detach().view(-1, self.reg_max), dim=-1),
                            reduction='batchmean'
                        )
                        loss_distill = distill_weight * (loss_distill_cls + loss_distill_reg) * batch_size
            #  总损失汇总
            opt_weight = 0.5 # alpha
            sar_weight = 1.0 # beta
            opt_loss = opt_weight * loss_opt
            sar_loss = sar_weight * loss_sar
            total_loss = loss_fus + opt_loss + sar_loss + loss_distill
            if self.training:
                save_dir = getattr(self.hyp, 'save_dir', '.')
                log_path = os.path.join(save_dir, 'aux_loss_metrics.csv')

                if not hasattr(self, '_log_step'):
                    self._log_step = 0

                file_exists = os.path.exists(log_path)
                with open(log_path, 'a') as f:
                    if not file_exists:
                        f.write('step,opt_loss,sar_loss,distill_loss,total_loss\n')

                    f.write(
                        f'{self._log_step},'
                        f'{opt_loss.item():.6f},'
                        f'{sar_loss.item():.6f},'
                        f'{loss_distill.item():.6f},'
                        f'{total_loss.item():.6f}\n'
                    )

                self._log_step += 1

            return total_loss, loss_items

        if isinstance(preds, (list, tuple)):
            # 提取用来算验证 Loss 的特征图部分
            # 根据 Detect 类的实现，特征图通常在第二个元素
            val_feats = preds[1] if len(preds) > 1 else preds[0]
            loss_fus, loss_items, _ = self.get_detailed_loss(val_feats, batch)
            return loss_fus, loss_items


    def get_detailed_loss(self, feats, batch):
        """
        它复制了父类 __call__ 的前半部分，并返回中间分配结果。
        """
        # 复制原生 __call__ 的处理逻辑
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 准备真值
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 预测解码
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # 运行 Assigner 拿到 fg_mask
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # 计算这一路本身的标准 Loss
        target_scores_sum = max(target_scores.sum(), 1)
        l_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        l_box, l_dfl = torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)

        if fg_mask.sum():
            target_bboxes_norm = target_bboxes / stride_tensor
            l_box, l_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes_norm, target_scores, target_scores_sum, fg_mask
            )

        # 包装 loss items
        loss_items = torch.stack([l_box, l_cls, l_dfl]).detach()
        main_loss = (l_box * self.hyp.box + l_cls * self.hyp.cls + l_dfl * self.hyp.dfl) * batch_size

        # 存储中间变量供蒸馏使用
        intermediate = {
            'fg_mask': fg_mask,
            'target_scores': target_scores,
            'target_bboxes': target_bboxes,
            'anchor_points': anchor_points,
            'stride_tensor': stride_tensor
        }

        return main_loss, loss_items, intermediate

    def slice_preds(self, feats):
        """辅助函数：切分预测值"""
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        return pred_distri.permute(0, 2, 1).contiguous(), \
               pred_scores.permute(0, 2, 1).contiguous()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items

# 旋转框loss
class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        # 使用旋转框专用的分配器
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        # 使用旋转框专用的bboxloss
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # feats 是解耦头的特征，pred_angle 是额外加的角度预测分支
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes 将 DFL 的距离和角度转回 (x, y, w, h, angle)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)
        # 为分配器准备预测框（乘以步长还原到真实像素尺度） 角度不缩放
        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

class MultiOBBLoss(v8OBBLoss):

    def __init__(self, model, use_distillation=False):
        self.model = model
        super().__init__(model)
        self.use_distillation = use_distillation
        self.training = model.training
        self.task = 'obb'


    def __call__(self, preds, batch):
        """
        preds:
            训练时来自 MultiOBB 的字典:
            {
                "fus": (feats_fus, angle_fus),
                "opt": (feats_opt, angle_opt),
                "sar": (feats_sar, angle_sar),
            }

            验证/推理时通常不是 dict，而是 fusion 路标准 OBB 输出

        batch:
            batch["bboxes"] 应为 OBB 的 xywhr, shape [N, 5]
        """
        if isinstance(preds, dict):
            self.training = self.model.training
            feats_fus, angle_fus = preds["fus"]
            batch_size = angle_fus.shape[0]
            device = angle_fus.device

            # 主分支详细损失（带中间变量）
            loss_fus, loss_items, intermediate_data = self.get_detailed_obb_loss(preds["fus"], batch)

            # 两个辅助分支，直接调用父类 v8OBBLoss
            loss_opt, _ = super().__call__(preds["opt"], batch)
            loss_sar, _ = super().__call__(preds["sar"], batch)

            loss_distill = torch.tensor(0.0, device=device)
            if self.use_distillation:
                # 提取中间变量
                fg_mask = intermediate_data['fg_mask']  # [B, Anchors] 前景掩码
                target_scores = intermediate_data['target_scores']  # [B, Anchors, NC] # 每个点的分类标签
                target_bboxes = intermediate_data['target_bboxes']  # [B, Anchors, 5]  # 每个格点的回归目标
                # 只选取每个目标中最匹配的一个格点进行学习，而不是所有正样本

                top1_mask = torch.zeros_like(fg_mask, dtype=torch.bool, device=device)
                anchor_scores = target_scores.max(-1)[0]  # TAL 计算出的融合分 (s^alpha * IoU^beta)

                with torch.no_grad():
                    for i in range(batch_size):
                        img_fg_mask = fg_mask[i]
                        if not img_fg_mask.any():
                            continue

                        # 找到当前图片所有正样本索引及对应的 GT 框
                        fg_idx = torch.where(img_fg_mask)[0]
                        img_gt_boxes = target_bboxes[i, fg_idx]   # [num_fg, 5]

                        # 利用 GT 框的唯一性区分不同实例 (同一目标的格点具有相同的 target_bboxes)
                        # 浮点 unique 有时不够稳，
                        unique_boxes, inverse_indices = torch.unique(
                            img_gt_boxes, dim=0, return_inverse=True
                        )

                        for instance_idx in range(len(unique_boxes)):
                            # 提取属于该实例的所有格点
                            instance_indices = fg_idx[inverse_indices == instance_idx]
                            # 在该实例内部，选出对齐得分最高的格点索引
                            best_idx = instance_indices[torch.argmax(anchor_scores[i, instance_indices])]
                            top1_mask[i, best_idx] = True

                    # 只对正样本点进行比拼和蒸馏
                    if top1_mask.any():
                        # 提取正样本点的数据  获取三路预测的原始 Logits (分类) 和分布 (回归)
                        dist_fus, score_fus, ang_fus = self.slice_preds(preds["fus"])
                        dist_opt, score_opt, ang_opt = self.slice_preds(preds["opt"])
                        dist_sar, score_sar, ang_sar = self.slice_preds(preds["sar"])
                        # 超参
                        teacher_threshold = 0.3  # 教师置信度阈值
                        T = 2.0
                        distill_weight = 0.05

                        with torch.no_grad():
                            # 计算老师是谁 (基于表现比拼)
                            qual_fus = (score_fus.sigmoid() * target_scores).sum(-1)
                            qual_opt = (score_opt.sigmoid() * target_scores).sum(-1)
                            qual_sar = (score_sar.sigmoid() * target_scores).sum(-1)
                            # teacher_mask: 1 代表 Optical 更好, 0 代表 SAR 更好
                            qual_fus_top1 = qual_fus[top1_mask]
                            qual_opt_top1 = qual_opt[top1_mask]
                            qual_sar_top1 = qual_sar[top1_mask]

                            max_qual_teacher, teacher_idx = torch.max(torch.stack([qual_opt_top1, qual_sar_top1]), dim=0) # [B, Anchors, 1]
                            valid_distill_mask = (max_qual_teacher > teacher_threshold) & (max_qual_teacher > qual_fus_top1)

                        if valid_distill_mask.any():
                            # 只取那些通过阈值筛选的样本进行计算
                            # teacher_mask: 1 代表 Optical 更优，0 代表 SAR 更优
                            teacher_mask_top1 = (
                                    qual_opt_top1[valid_distill_mask] > qual_sar_top1[valid_distill_mask]
                            ).float().view(-1,
                                           1)

                            logits_fus = score_fus[top1_mask][valid_distill_mask]
                            logits_opt = score_opt[top1_mask][valid_distill_mask]
                            logits_sar = score_sar[top1_mask][valid_distill_mask]

                            # 动态选择老师的分类 Logits
                            logits_teacher = teacher_mask_top1 * logits_opt.detach() \
                                             + (1.0 - teacher_mask_top1) * logits_sar.detach()

                            # 使用 KL 散度对齐概率分布 分类蒸馏
                            loss_distill_cls = F.kl_div(
                                F.log_softmax(logits_fus / T, dim=-1),
                                F.softmax(logits_teacher.detach() / T, dim=-1),
                                reduction='batchmean'
                            ) * (T ** 2)

                            d_fus = dist_fus[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)
                            d_opt = dist_opt[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)
                            d_sar = dist_sar[top1_mask][valid_distill_mask].view(-1, 4, self.reg_max)

                            teacher_mask_reg = teacher_mask_top1.view(-1, 1, 1)

                            # 动态选择老师的回归分布
                            d_teacher = teacher_mask_reg * d_opt.detach() \
                                        + (1.0 - teacher_mask_reg) * d_sar.detach()
                            # 回归蒸馏
                            loss_distill_reg = F.kl_div(
                                F.log_softmax(d_fus.view(-1, self.reg_max), dim=-1),
                                F.softmax(d_teacher.detach().view(-1, self.reg_max), dim=-1),
                                reduction='batchmean'
                            )
                            # angle 蒸馏
                            angle_fus_sel = ang_fus[top1_mask][valid_distill_mask]  # [M, 1]
                            angle_opt_sel = ang_opt[top1_mask][valid_distill_mask]
                            angle_sar_sel = ang_sar[top1_mask][valid_distill_mask]

                            angle_teacher = (
                                    teacher_mask_top1 * angle_opt_sel.detach()
                                    + (1.0 - teacher_mask_top1) * angle_sar_sel.detach()
                            )

                            # SmoothL1 会让-45和135的差距会比较大
                            # 用三角函数进行处理
                            # loss_distill_angle = F.smooth_l1_loss(angle_fus_sel, angle_teacher)
                            loss_distill_angle = F.mse_loss(torch.sin(angle_fus_sel), torch.sin(angle_teacher)) + \
                                                 F.mse_loss(torch.cos(angle_fus_sel), torch.cos(angle_teacher))
                            loss_distill = distill_weight * (
                                    loss_distill_cls + loss_distill_reg +  loss_distill_angle
                            )* batch_size

                total_loss = loss_fus + 0.5 * loss_opt + 1.0 * loss_sar + loss_distill

                if self.training:
                    log_path = 'aux_loss_metrics.csv'

                    if not hasattr(self, '_log_step'):
                        self._log_step = 0

                    is_new = not os.path.exists(log_path)
                    with open(log_path, 'a') as f:
                        if is_new:
                            f.write('step,opt_loss,sar_loss,distill_loss,total_loss\n')

                        f.write(
                            f'{self._log_step},'
                            f'{loss_opt.item():.6f},'
                            f'{loss_sar.item():.6f},'
                            f'{loss_distill.item():.6f},'
                            f'{total_loss.item():.6f}\n'
                        )

                    self._log_step += 1

                return total_loss, loss_items

            # 验证阶段走 fusion 路标准 OBB 输出
        if isinstance(preds, (list, tuple)):
            # preds[0] 是推理结果 [B, 300, 7]
            # preds[1] 是训练格式的输出 (feats, angle)
            # 我们调用 get_detailed_obb_loss 来计算真实的验证损失
            main_loss, loss_items, _ = self.get_detailed_obb_loss(preds[1], batch)

            # 验证阶段不需要计算蒸馏和辅助分支，只看主分支（Fusion路）的表现
            return main_loss / preds[1][0][0].shape[0], loss_items

    def get_detailed_obb_loss(self, pred, batch):
        """
        pred: 单路 OBB 预测，格式为 (feats, pred_angle)

        返回：
            main_loss,
            loss_items = [l_box, l_cls, l_dfl],
            intermediate = {
                fg_mask, target_scores, target_bboxes, anchor_points, stride_tensor
            }
        """
        feats, pred_angle = pred
        # --- 调试代码 ---
        for i, xi in enumerate(feats):
            if not isinstance(xi, torch.Tensor):
                print(f"!!! 发现非 Tensor 对象 !!! 索引: {i}, 类型: {type(xi)}, 内容: {xi}")
        # ----------------
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   # [B, A, nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   # [B, A, 4*reg_max]
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()     # [B, A, 1]

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        #  GT
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)),
                1,
            )

            rw = targets[:, 4] * imgsz[0].item()
            rh = targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # 过滤过小旋转框

            targets = self.preprocess(
                targets.to(self.device),
                batch_size,
                scale_tensor=imgsz[[1, 0, 1, 0]],  # 只缩放 xywh，不缩放 angle
            )

            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not an OBB dataset."
            ) from e

        # 解码旋转框
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # [B, A, 5]

        # assigner 只需要前4维乘 stride，角度不缩放
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        #  cls loss
        l_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # box + dfl loss
        l_box = torch.tensor(0.0, device=self.device)
        l_dfl = torch.tensor(0.0, device=self.device)

        if fg_mask.sum():
            target_bboxes = target_bboxes.clone()
            target_bboxes[..., :4] /= stride_tensor

            l_box, l_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        else:
            # 保持和官方 OBB loss 一致，确保 angle 分支有梯度图连接
            l_box += (pred_angle * 0).sum()

        loss_items = torch.stack([l_box, l_cls, l_dfl]).detach()
        main_loss = (
            l_box * self.hyp.box +
            l_cls * self.hyp.cls +
            l_dfl * self.hyp.dfl
        ) * batch_size

        intermediate = {
            "fg_mask": fg_mask,
            "target_scores": target_scores,
            "target_bboxes": target_bboxes,
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
        }

        return main_loss, loss_items, intermediate

    def slice_preds(self, pred):
        """
        pred: 单路 OBB 输出 (feats, pred_angle)

        返回：
            pred_distri: [B, A, 4*reg_max]
            pred_scores: [B, A, nc]
            pred_angle:  [B, A, 1]
        """
        feats, pred_angle = pred

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        return (
            pred_distri.permute(0, 2, 1).contiguous(),
            pred_scores.permute(0, 2, 1).contiguous(),
            pred_angle.permute(0, 2, 1).contiguous(),
        )


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps
