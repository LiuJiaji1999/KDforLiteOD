import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.checks import check_version
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.torch_utils import de_parallel

import functools
import math

# ------------------------------------------------------------
class LogicalLoss(nn.Module):
    def __init__(self, hyp, model, distiller='l2', task='detect') -> None:
        super().__init__()
        
        if distiller in ['l2', 'l1']:
            if task == 'detect':
                self.logical_loss = OutputLoss(hyp, distiller)
            elif task == 'pose':
                self.logical_loss = OutputLoss_Pose(hyp, distiller)
                self.logical_loss.kpt_shape = de_parallel(model).kpt_shape
            elif task == 'obb':
                self.logical_loss = OutputLoss_OBB(hyp, distiller)
            elif task == 'segment':
                self.logical_loss = OutputLoss_Seg(hyp, distiller)
        elif distiller == 'BCKD':
             self.logical_loss = BCKD(hyp)
        
        self.logical_loss.nc = de_parallel(model).nc
        self.logical_loss.no = de_parallel(model).model[-1].no
        self.logical_loss.reg_max = de_parallel(model).model[-1].reg_max
        self.logical_loss.stride = de_parallel(model).model[-1].stride
        self.logical_loss.use_dfl = self.logical_loss.reg_max > 1
        self.logical_loss.device = next(de_parallel(model).parameters()).device
        self.logical_loss.proj = torch.arange(de_parallel(model).model[-1].reg_max, dtype=torch.float, device=self.logical_loss.device)
        
        if distiller == 'BCKD':
            self.logical_loss.assigner = TaskAlignedAssigner(topk=10, num_classes=de_parallel(model).nc, alpha=0.5, beta=6.0)
            self.logical_loss.assigner_o2o = TaskAlignedAssigner(topk=1, num_classes=de_parallel(model).nc, alpha=0.5, beta=6.0)
        
        self.distiller = distiller
            
    def forward(self, s_p, t_p, batch, o2o=False):
        assert len(s_p) == len(t_p)
        if self.distiller == 'BCKD':
            loss = self.logical_loss(s_p, t_p, batch, o2o=o2o)
        else:
            loss = self.logical_loss(s_p, t_p, batch)
        return loss

# ------------------------------------------------------------
class OutputLoss(nn.Module):
    def __init__(self, hyp, distiller='l2'):
        super().__init__()
        
        if distiller == 'l2':
            box_loss = torch.nn.MSELoss(reduction='none')
            cls_loss = torch.nn.MSELoss(reduction='none')
        elif distiller == 'l1':
            box_loss = torch.nn.L1Loss(reduction='none')
            cls_loss = torch.nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError
        
        self.box_loss = box_loss
        self.cls_loss = cls_loss
        self.hyp = hyp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0].shape[0], self.no, -1) for xi in s_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0].shape[0], self.no, -1) for xi in t_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        
        return (lbox + lcls) * s_p[0].size(0)

class OutputLoss_Pose(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lpose = torch.zeros(1, device=self.device)  # pose loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lpose = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale.repeat(1, self.kpt_shape[0] * self.kpt_shape[1], 1))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lpose *= self.hyp.pose
        
        return (lbox + lcls + lpose) * s_p[0][0].size(0)

class OutputLoss_OBB(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobb = torch.zeros(1, device=self.device)  # obb loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lobb = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale)
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lobb *= self.hyp.box
        
        return (lbox + lcls + lobb) * s_p[0][0].size(0)

class OutputLoss_Seg(OutputLoss):
    def __init__(self, hyp, distiller='l2'):
        super().__init__(hyp, distiller)
    
    def forward(self, s_p, t_p, batch):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lseg = torch.zeros(1, device=self.device)  # seg loss
        lproto = torch.zeros(1, device=self.device) # proto loss
        
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0][0].shape[0], self.no, -1) for xi in s_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0][0].shape[0], self.no, -1) for xi in t_p[0]], 2).split(
            (self.reg_max * 4, self.nc), 1)
        
        t_obj_scale = t_pred_scores.sigmoid().max(1)[0].unsqueeze(1)
        
        lbox = torch.mean(self.box_loss(s_pred_distri, t_pred_distri) * t_obj_scale.repeat(1, self.reg_max * 4, 1))
        lcls = torch.mean(self.cls_loss(s_pred_scores, t_pred_scores) * t_obj_scale.repeat(1, self.nc, 1))
        lseg = torch.mean(self.cls_loss(s_p[1], t_p[1]) * t_obj_scale.repeat(1, s_p[1].size(1), 1))
        lproto = torch.mean(self.cls_loss(s_p[2], t_p[2]))
        
        lbox *= self.hyp.dfl
        lcls *= self.hyp.cls
        lseg *= self.hyp.box
        lproto *= self.hyp.box
        
        return (lbox + lcls + lseg + lproto) * s_p[0][0].size(0)

# ------------------------------------------------------------
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
                      T * T)

    return kd_loss

class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd

@weighted_loss
def novel_kd_loss(pred,
                  soft_label,
                  detach_target=True,
                  beta=1.0):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = soft_label.sigmoid()
    score = pred.sigmoid()

    if detach_target:
        target = target.detach()

    scale_factor = target - score
    kd_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor.abs().pow(beta)
    kd_loss = kd_loss.sum(dim=-1, keepdim=False)
    return kd_loss

class NovelKDLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10, threshold=0.05):
        super(NovelKDLoss, self).__init__()
        # assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T
        self.threshold = threshold

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                beta=1.0):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * novel_kd_loss(pred, soft_label, weight, reduction=reduction, avg_factor=avg_factor, beta=beta)

        return loss_kd

class BCKD(nn.Module):
    def __init__(self, hyp) -> None:
        super().__init__()
        
        self.hyp = hyp
        
        self.loss_score_kd_ratio = 7.5
        self.loss_kd = NovelKDLoss(loss_weight=1.0)
        self.loss_ld = KnowledgeDistillationKLDivLoss(loss_weight=0.0, T=10)
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def forward(self, s_p, t_p, batch, o2o=False):
        s_pred_distri, s_pred_scores = torch.cat([xi.view(s_p[0].shape[0], self.no, -1) for xi in s_p], 2).split(
            (self.reg_max * 4, self.nc), 1)
        t_pred_distri, t_pred_scores = torch.cat([xi.view(t_p[0].shape[0], self.no, -1) for xi in t_p], 2).split(
            (self.reg_max * 4, self.nc), 1)

        s_pred_distri, s_pred_scores = s_pred_distri.permute(0, 2, 1).contiguous(), s_pred_scores.permute(0, 2, 1).contiguous()
        t_pred_distri, t_pred_scores = t_pred_distri.permute(0, 2, 1).contiguous(), t_pred_scores.permute(0, 2, 1).contiguous()
        
        dtype = s_pred_scores.dtype
        batch_size = s_pred_scores.shape[0]
        imgsz = torch.tensor(s_p[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(s_p, self.stride, 0.5)
        
        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        s_pred_bboxes, t_pred_bboxes = self.bbox_decode(anchor_points, s_pred_distri), self.bbox_decode(anchor_points, t_pred_distri)
        
        if o2o:
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner_o2o(
                s_pred_scores.detach().sigmoid(), (s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)  
        else:
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                    s_pred_scores.detach().sigmoid(), (s_pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        
        weight_targets1 = s_pred_scores.detach().sigmoid()
        weight_targets2 = t_pred_scores.detach().sigmoid()
        weight_targets3 = weight_targets2 - weight_targets1
        total_weight_targets = weight_targets3.abs().pow(1.0).max(dim=-1)[0] * fg_mask
        
        loss_score_kd = ((1 - bbox_iou(s_pred_bboxes, t_pred_bboxes.detach(), xywh=False, CIoU=True).squeeze()) * total_weight_targets).sum() / target_scores_sum
        loss_score_kd *= self.loss_score_kd_ratio
        
        fg_mask = fg_mask.bool()
        weight_targets = s_pred_scores.detach().sigmoid() + 1e-4
        weight_targets = weight_targets.max(dim=-1)[0][fg_mask] + 1e-4
        s_pred_corners, t_pred_corners = s_pred_distri[fg_mask].reshape(-1, self.reg_max), t_pred_distri[fg_mask].reshape(-1, self.reg_max)
        loss_ld = self.loss_ld(s_pred_corners, t_pred_corners, weight_targets[:, None].expand(-1, 4).reshape(-1), 4.0)
        loss_kd = self.loss_kd(s_pred_scores, t_pred_scores, fg_mask, target_scores_sum)
        
        return loss_score_kd + loss_ld + loss_kd

class PKDLoss(nn.Module):
    """Feature PKD-style loss: normalize each feature map then minimize MSE."""
    def __init__(self, channels_s=None, channels_t=None, eps=1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return x.view(n, c, h, w)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        for s, t in zip(y_s, y_t):
            assert s.shape == t.shape
            losses.append(F.mse_loss(self._norm(s), self._norm(t)))
        return sum(losses)


def _build_feature_distiller(distiller, channels_s, channels_t):
    distiller = str(distiller).lower()
    if distiller == 'mimic':
        return MimicLoss(channels_s, channels_t)
    if distiller == 'mgd':
        return MGDLoss(channels_s, channels_t)
    if distiller == 'cwd':
        return CWDLoss(channels_s, channels_t)
    if distiller == 'chsim':
        return ChSimLoss(channels_s, channels_t)
    if distiller == 'sp':
        return SPKDLoss(channels_s, channels_t)
    if distiller == 'pkd':
        return PKDLoss(channels_s, channels_t)
    if distiller == 'sl2':
        return SL2Loss(channels_s, channels_t)
    raise NotImplementedError(f'Unknown feature distiller: {distiller}')


class FeatureLoss(nn.Module):
    """
    Feature KD with optional layer-class adaptive selection and historical prototype review.

    Legacy mode:
        FeatureLoss(..., distiller='cwd') behaves like the original implementation.

    Proposed mode:
        Set feature_loss_type='adaptive_review'. The base feature distiller is selected by
        adapt_base_loss_type, e.g. cwd/mgd/mimic/chsim/sp/pkd.
    """
    def __init__(self,
                 channels_s,
                 channels_t,
                 distiller='cwd',
                 hyp=None,
                 nc=None):
        super().__init__()
        assert len(channels_s) == len(channels_t), 'student and teacher feature layers must be paired.'

        self.hyp = hyp
        self.nc = int(nc if nc is not None else getattr(hyp, 'nc', 80))
        self.nl = len(channels_t)
        self.eps = 1e-6
        self.last_feature_loss = None
        self.last_review_loss = None
        self.last_adapt_weight = None
        self.last_review_lambda = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0)
            if channel != tea_channel else nn.Identity()
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        # Register normalization layers correctly; affine=False keeps no extra trainable params.
        self.norm = nn.ModuleList([
            nn.BatchNorm2d(tea_channel, affine=False) for tea_channel in channels_t
        ])

        self.distiller_name = str(distiller).lower()
        self.use_adaptive_review = self.distiller_name in ['adaptive_review', 'arfd', 'arkd']
        self.base_distiller_name = str(getattr(hyp, 'adapt_base_loss_type', 'cwd')).lower() if self.use_adaptive_review else self.distiller_name

        self.feature_loss = _build_feature_distiller(self.base_distiller_name, channels_t, channels_t)
        self.layer_feature_loss = nn.ModuleList([
            _build_feature_distiller(self.base_distiller_name, [c], [c]) for c in channels_t
        ])

        # Module switches for clean ablation.
        self.adapt_enable = bool(getattr(hyp, 'adapt_enable', True))
        self.review_enable = bool(getattr(hyp, 'review_enable', True))
        self.use_teacher_quality = bool(getattr(hyp, 'adapt_use_teacher_quality', True))
        self.use_student_demand = bool(getattr(hyp, 'adapt_use_student_demand', True))
        self.use_R = bool(getattr(hyp, 'adapt_use_R', True))
        self.use_Q = bool(getattr(hyp, 'adapt_use_Q', True))
        self.use_U = bool(getattr(hyp, 'adapt_use_U', True))
        self.use_Sim = bool(getattr(hyp, 'adapt_use_Sim', True))
        self.use_G = bool(getattr(hyp, 'adapt_use_G', True))
        self.use_review_consistency = bool(getattr(hyp, 'review_use_consistency', True))
        self.update_memory = bool(getattr(hyp, 'review_update_memory', True))

        self.tau_gap = float(getattr(hyp, 'adapt_tau_gap', 1.0))
        self.mu_gap = 1.0
        self.sigma_gap = 0.5
        self.entropy_tau = float(getattr(hyp, 'adapt_entropy_tau', 1.0))
        self.adapt_min_weight = float(getattr(hyp, 'adapt_min_weight', 0.25))
        self.adapt_max_weight = float(getattr(hyp, 'adapt_max_weight', 2.0))
        self.adapt_norm_weight = bool(getattr(hyp, 'adapt_norm_weight', True))
        self.review_momentum = float(getattr(hyp, 'review_momentum', 0.999))
        self.review_quality_thr = float(getattr(hyp, 'review_quality_thr', 0.5))
        self.review_ratio = float(getattr(hyp, 'review_ratio', 1.0))
        self.review_start_epoch = int(getattr(hyp, 'review_start_epoch', 0))
        self.review_schedule = str(getattr(hyp, 'review_schedule', 'sine')).lower()

        self.adapt_global_loss_ratio = float(getattr(hyp, 'adapt_global_loss_ratio', 0.1))
        self.adapt_region_loss_ratio = float(getattr(hyp, 'adapt_region_loss_ratio', 1.0))


        # Historical teacher prototypes M_{l,c}; one memory matrix per layer because C differs by layer.
        for li, c in enumerate(channels_t):
            self.register_buffer(f'review_memory_{li}', torch.zeros(self.nc, c))
            self.register_buffer(f'review_valid_{li}', torch.zeros(self.nc, dtype=torch.bool))
    
    def _get_review_memory(self, li, device):
        """
        Return historical review memory on the same device as current features.

        review_memory_{li} stores historical teacher prototypes M_{l,c}.
        review_valid_{li} marks whether class c already has a valid prototype.

        This function is a safety guard for cases where buffers are still on CPU
        after module initialization or checkpoint loading.
        """
        mem_name = f'review_memory_{li}'
        valid_name = f'review_valid_{li}'

        mem = getattr(self, mem_name)
        valid = getattr(self, valid_name)

        if mem.device != device:
            mem = mem.to(device)
            self._buffers[mem_name] = mem

        if valid.device != device:
            valid = valid.to(device)
            self._buffers[valid_name] = valid

        return mem, valid

    def _align_and_norm(self, y_s, y_t):
        tea_feats, stu_feats = [], []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t.detach())
            tea_feats.append(t)
            stu_feats.append(s)
        return stu_feats, tea_feats

    def _spatial_standardize(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        return (x - mean) / (std + self.eps)

    def _flatten_pred_maps(self, pred):
        maps = []
        if pred is None:
            return maps
        if isinstance(pred, dict):
            # YOLOv10-like heads may return one2many/one2one dictionaries.
            pred = pred.get('one2many', pred.get('one2one', pred))
        if isinstance(pred, torch.Tensor):
            return [pred] if pred.ndim == 4 else []
        if isinstance(pred, (list, tuple)):
            for x in pred:
                maps.extend(self._flatten_pred_maps(x))
        return maps

    def _match_pred_map(self, pred_maps, h, w):
        for p in pred_maps:
            if isinstance(p, torch.Tensor) and p.ndim == 4 and p.shape[2] == h and p.shape[3] == w:
                return p.detach()
        return None

    def _decode_layer_boxes(self, p):
        """Decode YOLO-like DFL distances in feature-cell coordinates for teacher localization quality Q."""
        if p is None or p.ndim != 4 or p.shape[1] <= self.nc:
            return None
        b, no, h, w = p.shape
        reg_ch = no - self.nc
        if reg_ch <= 0 or reg_ch % 4 != 0:
            return None
        reg_max = reg_ch // 4
        dist = p[:, :reg_ch].float().view(b, 4, reg_max, h, w).softmax(2)
        proj = torch.arange(reg_max, device=p.device, dtype=dist.dtype).view(1, 1, reg_max, 1, 1)
        dist = (dist * proj).sum(2)  # [B,4,H,W], l,t,r,b in cell units
        yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing='ij')
        xx = xx.float() + 0.5
        yy = yy.float() + 0.5
        x1 = xx[None] - dist[:, 0]
        y1 = yy[None] - dist[:, 1]
        x2 = xx[None] + dist[:, 2]
        y2 = yy[None] + dist[:, 3]
        return torch.stack((x1, y1, x2, y2), dim=1)  # [B,4,H,W]

    def _box_iou_vec(self, boxes, gt):
        # boxes: [N,4], gt: [4] in xyxy.
        lt = torch.maximum(boxes[:, :2], gt[:2])
        rb = torch.minimum(boxes[:, 2:], gt[2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        area2 = (gt[2] - gt[0]).clamp(min=0) * (gt[3] - gt[1]).clamp(min=0)
        return inter / (area1 + area2 - inter + self.eps)

    def _project_box(self, xywh, h, w, device):
        # batch['bboxes'] is normalized xywh in Ultralytics detection dataloader.
        x, y, bw, bh = xywh
        x1 = (x - bw / 2.0) * w
        y1 = (y - bh / 2.0) * h
        x2 = (x + bw / 2.0) * w
        y2 = (y + bh / 2.0) * h
        fx1, fy1, fx2, fy2 = x1.clamp(0, w), y1.clamp(0, h), x2.clamp(0, w), y2.clamp(0, h)
        ix1 = int(torch.floor(fx1).item())
        iy1 = int(torch.floor(fy1).item())
        ix2 = int(torch.ceil(fx2).item())
        iy2 = int(torch.ceil(fy2).item())
        # Guarantee at least one positive feature point for small objects.
        if ix2 <= ix1:
            cx = int(torch.clamp(((fx1 + fx2) / 2).round(), 0, w - 1).item())
            ix1, ix2 = cx, min(cx + 1, w)
        if iy2 <= iy1:
            cy = int(torch.clamp(((fy1 + fy2) / 2).round(), 0, h - 1).item())
            iy1, iy2 = cy, min(cy + 1, h)
        gt = torch.tensor([fx1, fy1, fx2, fy2], device=device, dtype=torch.float32)
        return ix1, iy1, ix2, iy2, gt

    def _review_lambda(self, epoch, total_epochs, device):
        if epoch is None or total_epochs is None or total_epochs <= 1:
            val = 0.0 if self.review_schedule == 'sine' else 1.0
        elif epoch < self.review_start_epoch:
            val = 0.0
        elif self.review_schedule == 'constant':
            val = 1.0 #0.5
        elif self.review_schedule == 'linear':
            val = min(max((epoch - self.review_start_epoch) / max(total_epochs - self.review_start_epoch - 1, 1), 0.0), 1.0)
        else:
            progress = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
            val = math.sin(math.pi * progress)
        return torch.tensor(float(val), device=device)

    def _teacher_quality(self, R, Q, U):
        if not self.use_teacher_quality:
            return torch.ones_like(R)
        terms, weights = [], []
        if self.use_R:
            terms.append(R); weights.append(0.4)
        if self.use_Q:
            terms.append(Q); weights.append(0.4)
        if self.use_U:
            terms.append(1.0 - U); weights.append(0.2)
        if not terms:
            return torch.ones_like(R)
        return sum(w * t for w, t in zip(weights, terms)) / sum(weights)

    def _student_demand(self, Sim, G):
        if not self.use_student_demand:
            return torch.ones_like(Sim)
        if self.use_Sim and self.use_G:
            # return 0.5 * ((1.0 - Sim) + G)
            return (1.0 - Sim) * G
        if self.use_Sim:
            return 1.0 - Sim
        if self.use_G:
            return G
        return torch.ones_like(Sim)

    def _adaptive_review_forward(self, stu_feats, tea_feats, batch, t_pred, epoch, total_epochs):
        device = stu_feats[0].device
        pred_maps = self._flatten_pred_maps(t_pred)
        batch_idx = batch.get('batch_idx', None) if isinstance(batch, dict) else None
        gt_cls = batch.get('cls', None) if isinstance(batch, dict) else None
        gt_boxes = batch.get('bboxes', None) if isinstance(batch, dict) else None
        if batch_idx is None or gt_cls is None or gt_boxes is None or gt_boxes.numel() == 0:
            base_loss = self.feature_loss(stu_feats, tea_feats)
            z = base_loss.detach() * 0.0
            self.last_feature_loss, self.last_review_loss = base_loss.detach(), z
            self.last_adapt_weight, self.last_review_lambda = torch.tensor(1.0, device=device), z
            return base_loss

        batch_idx = batch_idx.to(device).long().view(-1)
        gt_cls = gt_cls.to(device).long().view(-1)
        gt_boxes = gt_boxes.to(device).float()
        lambda_re = self._review_lambda(epoch, total_epochs, device)

        feature_losses, review_losses, all_A = [], [], []
        for li, (s, t) in enumerate(zip(stu_feats, tea_feats)):
            B, C, H, W = t.shape
            pmap = self._match_pred_map(pred_maps, H, W)
            cls_logits = pmap[:, -self.nc:].float() if pmap is not None and pmap.shape[1] >= self.nc else None
            box_map = self._decode_layer_boxes(pmap) if pmap is not None else None

            class_masks = {}
            stat = {}
            for gi in range(gt_boxes.shape[0]):
                b = int(batch_idx[gi].item())
                c = int(gt_cls[gi].item())
                if b < 0 or b >= B or c < 0 or c >= self.nc:
                    continue
                ix1, iy1, ix2, iy2, gt_xyxy = self._project_box(gt_boxes[gi], H, W, device)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if c not in class_masks:
                    class_masks[c] = torch.zeros(B, H, W, dtype=torch.bool, device=device)
                    stat[c] = dict(R_sum=torch.tensor(0.0, device=device),
                                   U_sum=torch.tensor(0.0, device=device),
                                   Q_sum=torch.tensor(0.0, device=device),
                                   n=torch.tensor(0.0, device=device))
                class_masks[c][b, iy1:iy2, ix1:ix2] = True
                n = float((iy2 - iy1) * (ix2 - ix1))
                stat[c]['n'] = stat[c]['n'] + n

                if cls_logits is not None:
                    region_logits = cls_logits[b, :, iy1:iy2, ix1:ix2]
                    cls_prob = region_logits[c].sigmoid()
                    stat[c]['R_sum'] = stat[c]['R_sum'] + cls_prob.sum()
                    prob = F.softmax(region_logits.flatten(1).transpose(0, 1) / self.entropy_tau, dim=1)
                    entropy = -(prob * (prob + self.eps).log()).sum(1) / math.log(max(self.nc, 2))
                    stat[c]['U_sum'] = stat[c]['U_sum'] + entropy.sum()
                else:
                    stat[c]['R_sum'] = stat[c]['R_sum'] + n

                if box_map is not None:
                    boxes = box_map[b, :, iy1:iy2, ix1:ix2].permute(1, 2, 0).reshape(-1, 4)
                    stat[c]['Q_sum'] = stat[c]['Q_sum'] + self._box_iou_vec(boxes.float(), gt_xyxy).sum()
                elif cls_logits is not None:
                    # Safe proxy when decoded boxes are unavailable: confidence also reflects localization tendency.
                    stat[c]['Q_sum'] = stat[c]['Q_sum'] + cls_prob.detach().sum()
                else:
                    stat[c]['Q_sum'] = stat[c]['Q_sum'] + n

            if not class_masks:
                feature_losses.append(self.layer_feature_loss[li]([s], [t]))
                continue

            s_std, t_std = self._spatial_standardize(s), self._spatial_standardize(t)
            layer_A, layer_review = [], []
            region_feature_losses = []
            for c, mask in class_masks.items():
                if mask.sum() <= 0:
                    continue
                sp = s.permute(0, 2, 3, 1)[mask]
                tp = t.permute(0, 2, 3, 1)[mask]
                s_proto = sp.mean(0)
                t_proto = tp.mean(0).detach()

                sim = (F.cosine_similarity(s_proto[None], t_proto[None], dim=1).squeeze(0) + 1.0) * 0.5
                sdp = s_std.permute(0, 2, 3, 1)[mask]
                tdp = t_std.permute(0, 2, 3, 1)[mask]
                D = (sdp - tdp).abs().mean()
                mu_g = getattr(self, 'mu_gap', 1.0)
                sigma_g = max(getattr(self, 'sigma_gap', 0.5), self.eps)
                G = torch.exp(-((D - mu_g) ** 2) / (2.0 * sigma_g ** 2))
                # G = torch.exp(-D / max(self.tau_gap, self.eps))

                n = stat[c]['n'].clamp(min=1.0)
                R = (stat[c]['R_sum'] / n).clamp(0.0, 1.0)
                U = (stat[c]['U_sum'] / n).clamp(0.0, 1.0) if cls_logits is not None else torch.tensor(0.0, device=device)
                Q = (stat[c]['Q_sum'] / n).clamp(0.0, 1.0)
                KT = self._teacher_quality(R, Q, U).clamp(0.0, 1.0)
                KS = self._student_demand(sim.clamp(0.0, 1.0), G.clamp(0.0, 1.0)).clamp(0.0, 1.0)
                A = (KT * KS).detach() if self.adapt_enable else torch.ones((), device=device)
                layer_A.append(A)
                all_A.append(A)

                # Region-level auxiliary feature KD makes A_{l,c} directly act on class foreground prototypes.
                region_feature_losses.append(A * F.mse_loss(sp, tp.detach()))
                # ------------------------------------------------------------
                # Review loss: use the historical teacher prototype M^-_{l,c}
                # before updating the memory with the current teacher prototype.
                # ------------------------------------------------------------
                mem,valid = self._get_review_memory(li, device=device)
                if self.review_enable and bool(valid[c].item()):
                     # detach(): historical memory is a fixed target, not optimized by gradients.
                    # clone(): avoid autograd error because mem[c] will be updated in-place later.
                    # to(): make sure memory target is on the same device as current prototypes.
                    m_proto = mem[c].detach().clone().to(device)
                    # Historical consistency r_{l,c}.
                    # A high value means the current teacher prototype is consistent with
                    # the historical teacher prototype, so review is more reliable.
                    rr = (F.cosine_similarity(
                        t_proto.detach()[None], 
                        m_proto[None], 
                        dim=1).squeeze(0) + 1.0
                        ) * 0.5
                    if not self.use_review_consistency:
                        rr = torch.ones_like(rr)
                        # A and r are weighting factors, so both are detached.
                    # Only s_proto receives gradients from the review constraint.
                    layer_review.append(A.detach() * rr.detach() * F.mse_loss(s_proto, m_proto))
                # ------------------------------------------------------------
                # Memory update: update M^-_{l,c} to M^+_{l,c} after review loss.
                # Only high-quality teacher prototypes are written into memory.
                # ------------------------------------------------------------
                if self.review_enable and self.update_memory:
                    with torch.no_grad():
                        if bool((KT > self.review_quality_thr).item()):
                            t_proto_detached = t_proto.detach().to(device)

                            if bool(valid[c].item()):
                                mem[c].mul_(self.review_momentum).add_(
                                    t_proto_detached, 
                                    alpha=1.0 - self.review_momentum)
                            else:
                                mem[c].copy_(t_proto_detached)
                                valid[c] = True

            A_tensor = torch.stack(layer_A) if layer_A else torch.ones(1, device=device)
            if not self.adapt_enable:
                layer_weight = torch.ones((), device=device)
            else:
                if self.adapt_norm_weight and A_tensor.numel() > 0:
                    # Bounded affine mapping keeps gradients stable while preserving the direction of A_{l,c}:
                    # low A weakens unreliable/unneeded KD, high A strengthens reliable and needed KD.
                    layer_weight = self.adapt_min_weight + \
                        (self.adapt_max_weight - self.adapt_min_weight) * A_tensor.mean()
                else:
                    layer_weight = A_tensor.mean()

                layer_weight = layer_weight.clamp(
                    self.adapt_min_weight,
                    self.adapt_max_weight
                ).detach()

            # Keep the original base distiller as the main signal; add a small A-weighted foreground term.
            base_layer_loss = self.layer_feature_loss[li]([s], [t])
            fg_loss = sum(region_feature_losses) / max(len(region_feature_losses), 1) if region_feature_losses else base_layer_loss * 0.0
            fg_ratio = float(getattr(self.hyp, 'adapt_region_loss_ratio', 0.25))
            feature_losses.append(base_layer_loss * layer_weight + fg_ratio * fg_loss)
            if layer_review:
                review_losses.append(sum(layer_review) / len(layer_review))

        feature_loss = sum(feature_losses) if feature_losses else stu_feats[0].sum() * 0.0
        review_loss = sum(review_losses) / len(review_losses) if review_losses else feature_loss.detach() * 0.0
        total = feature_loss + self.review_ratio * lambda_re * review_loss

        self.last_feature_loss = feature_loss.detach()
        self.last_review_loss = (self.review_ratio * lambda_re * review_loss).detach()
        self.last_review_lambda = lambda_re.detach()
        self.last_adapt_weight = torch.stack(all_A).mean().detach() if all_A else torch.tensor(1.0, device=device)
        return total

    def forward(self, y_s, y_t, batch=None, t_pred=None, epoch=None, total_epochs=None):
        assert len(y_s) == len(y_t)
        stu_feats, tea_feats = self._align_and_norm(y_s, y_t)
        if not self.use_adaptive_review:
            loss = self.feature_loss(stu_feats, tea_feats)
            self.last_feature_loss = loss.detach()
            self.last_review_loss = loss.detach() * 0.0
            self.last_adapt_weight = torch.tensor(1.0, device=loss.device)
            self.last_review_lambda = loss.detach() * 0.0
            return loss
        if batch is None:
            # Safe fallback for warmup shape probing or legacy calls.
            loss = self.feature_loss(stu_feats, tea_feats)
            self.last_feature_loss = loss.detach()
            self.last_review_loss = loss.detach() * 0.0
            self.last_adapt_weight = torch.tensor(1.0, device=loss.device)
            self.last_review_lambda = loss.detach() * 0.0
            return loss
        return self._adaptive_review_forward(stu_feats, tea_feats, batch, t_pred, epoch, total_epochs)

# ------------------------------------------------------------
class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss

# ------------------------------------------------------------
class MGDLoss(nn.Module):
    def __init__(self,
                 channels_s,
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ])

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

# ------------------------------------------------------------
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau,
                                       dim=1)  # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (
                    self.tau**2)
            # cost = torch.sum(-softmax_pred_T * logsoftmax(s.view(-1, W * H)/self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss

# ------------------------------------------------------------
class ChSimLoss(nn.Module):
    '''
    A loss module for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html
    '''
    def __init__(self, channels_s, channels_t) -> None:
        super().__init__()
    
    @staticmethod
    def batch_loss(f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)
        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        g_diff = emd_s - emd_t
        loss = (g_diff * g_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss
    
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.batch_loss(s, t) / s.size(0))
        loss = sum(losses)
        return loss

# ------------------------------------------------------------
class SPKDLoss(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self, channels_s, channels_t):
        super(SPKDLoss, self).__init__()

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return F.normalize(torch.matmul(z, torch.t(z)), 1)

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            g_t = self.matmul_and_normalize(t)
            g_s = self.matmul_and_normalize(s)

            sp_loss = torch.norm(g_t - g_s) ** 2
            sp_loss = sp_loss.sum() / s.size(0)
            losses.append(sp_loss)
        loss = sum(losses)
        return loss

# ------------------------------------------------------------
class SL2Loss(nn.Module):
    """
    Score-weighted L2 distillation loss.

    This implementation follows the original YOLO SL2 idea:
    teacher confidence scores are used as spatial weights for feature L2 loss.

    It is rewritten to be consistent with other distillation losses such as
    MGDLoss, CWDLoss, ChSimLoss and SPKDLoss:
        loss = SL2Loss(channels_s, channels_t)
        loss(y_s, y_t)

    Args:
        channels_s (list): Student feature channels. Kept for interface consistency.
        channels_t (list): Teacher feature channels. Kept for interface consistency.
        eps (float): Small value for numerical stability.
        use_teacher_score (bool): Whether to use teacher score maps if provided.
    """

    def __init__(self,
                 channels_s,
                 channels_t,
                 eps=1e-9,
                 use_teacher_score=True):
        super(SL2Loss, self).__init__()
        self.eps = eps
        self.use_teacher_score = use_teacher_score

    def forward(self, y_s, y_t, teacher_scores=None):
        """
        Forward computation.

        Args:
            y_s (list[Tensor]): Student features, each with shape (N, C, H, W).
            y_t (list[Tensor]): Teacher features, each with shape (N, C, H, W).
            teacher_scores (tuple/list[Tensor], optional):
                Teacher score maps for each feature level.
                Each score map should have shape (N, 1, H*W) or (N, 1, H, W).

        Returns:
            Tensor: Distillation loss over all feature levels.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            if self.use_teacher_score and teacher_scores is not None:
                score = teacher_scores[idx]
            else:
                score = self._build_score_from_teacher_feature(t)

            losses.append(self.get_dis_loss(s, t, score))

        return sum(losses)

    def get_dis_loss(self, preds_S, preds_T, teacher_score):
        """
        Compute score-weighted L2 loss for one feature level.

        Args:
            preds_S (Tensor): Student feature, shape (N, C, H, W).
            preds_T (Tensor): Teacher feature, shape (N, C, H, W).
            teacher_score (Tensor): Spatial weight, shape (N, 1, H*W) or (N, 1, H, W).

        Returns:
            Tensor: Score-weighted L2 loss.
        """
        n, c, h, w = preds_S.shape

        student_feat = preds_S.view(n, c, -1)
        teacher_feat = preds_T.detach().view(n, c, -1)

        teacher_score = self._format_teacher_score(
            teacher_score=teacher_score,
            n=n,
            h=h,
            w=w,
            device=preds_S.device,
            dtype=preds_S.dtype
        )

        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + self.eps)

        return weighted_mse

    def _format_teacher_score(self, teacher_score, n, h, w, device, dtype):
        """
        Format teacher score to shape (N, 1, H*W).
        """
        teacher_score = teacher_score.to(device=device, dtype=dtype)

        if teacher_score.dim() == 4:
            teacher_score = teacher_score.view(n, 1, -1)
        elif teacher_score.dim() == 3:
            pass
        else:
            raise ValueError(
                f"teacher_score should have shape (N, 1, H, W) or (N, 1, H*W), "
                f"but got {teacher_score.shape}"
            )

        if teacher_score.shape[-1] != h * w:
            teacher_score = teacher_score.view(n, 1, int(teacher_score.shape[-1] ** 0.5), -1)
            teacher_score = F.interpolate(
                teacher_score,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).view(n, 1, -1)

        return teacher_score.clamp(min=0.0)

    def _build_score_from_teacher_feature(self, teacher_feat):
        """
        Build a fallback spatial score map from teacher feature energy.

        This is used when external teacher confidence scores are not provided.
        It keeps the loss callable as loss(y_s, y_t), like MGD/CWD/SPKD/ChSim.
        """
        with torch.no_grad():
            n, c, h, w = teacher_feat.shape

            score = teacher_feat.detach().pow(2).mean(dim=1, keepdim=True)
            score = score.view(n, 1, -1)

            score = score / (score.mean(dim=-1, keepdim=True) + self.eps)

        return score
