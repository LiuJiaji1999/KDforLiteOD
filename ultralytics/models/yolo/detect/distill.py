# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys, os, torch, math, time, warnings
import matplotlib
matplotlib.use('AGG')
import matplotlib.pylab as plt
import torch.nn as nn
from torch import optim
from thop import clever_format
from functools import partial
from torch import distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime

from copy import copy, deepcopy
from pathlib import Path

import torch.nn.functional as F

import numpy as np

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, yaml_save, callbacks, __version__
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.checks import check_imgsz, print_args, check_amp
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, one_cycle, init_seeds, select_device
from ultralytics.utils.distill_loss import LogicalLoss, FeatureLoss
from ultralytics.nn.extra_modules.kernel_warehouse import get_temperature

class AdaptiveReviewKD(nn.Module):
    """
    Adaptive Knowledge Selection + Historical Teacher Prototype Review.

    支持：
    1. 模块间消融：
        - use_aks
        - use_review
        - use_proto_feature_kd

    2. AKS 模块内消融：
        - aks_use_r
        - aks_use_q
        - aks_use_sim
        - aks_use_g
        - aks_use_u

    3. Review 模块内消融：
        - review_use_adapt_gate
        - review_use_consistency
        - review_update_with_aks
        - review_theta_h
        - review_momentum
        - review_lambda_schedule
    """

    def __init__(
        self,
        nc,

        # ------------------------------
        # Module-level ablation
        # ------------------------------
        use_aks=True,
        use_review=True,
        use_proto_feature_kd=True,

        # ------------------------------
        # AKS component ablation
        # ------------------------------
        aks_use_r=True,
        aks_use_q=True,
        aks_use_sim=True,
        aks_use_g=True,
        aks_use_u=True,

        aks_w_r=0.24,
        aks_w_q=0.24,
        aks_w_sim=0.20,
        aks_w_g=0.20,
        aks_w_u=0.12,

        aks_tau_g=1.0,
        aks_detach=True,
        aks_renorm_weights=True,

        # 当前代码中如果没有 teacher box decode，则 Q_l,c 用 R_l,c 近似
        use_q_fallback=True,

        # ------------------------------
        # Review ablation
        # ------------------------------
        review_use_adapt_gate=True,
        review_use_consistency=True,
        review_update_with_aks=True,
        review_theta_h=0.5,
        review_momentum=0.999,
        review_lambda_schedule="sine",

        eps=1e-6,
    ):
        super().__init__()

        self.nc = int(nc)

        # module-level switches
        self.use_aks = use_aks
        self.use_review = use_review
        self.use_proto_feature_kd = use_proto_feature_kd

        # AKS component switches
        self.aks_use_r = aks_use_r
        self.aks_use_q = aks_use_q
        self.aks_use_sim = aks_use_sim
        self.aks_use_g = aks_use_g
        self.aks_use_u = aks_use_u

        # AKS weights
        self.aks_w_r = aks_w_r
        self.aks_w_q = aks_w_q
        self.aks_w_sim = aks_w_sim
        self.aks_w_g = aks_w_g
        self.aks_w_u = aks_w_u

        self.aks_tau_g = aks_tau_g
        self.aks_detach = aks_detach
        self.aks_renorm_weights = aks_renorm_weights
        self.use_q_fallback = use_q_fallback

        # Review switches
        self.review_use_adapt_gate = review_use_adapt_gate
        self.review_use_consistency = review_use_consistency
        self.review_update_with_aks = review_update_with_aks
        self.review_theta_h = review_theta_h
        self.review_momentum = review_momentum
        self.review_lambda_schedule = review_lambda_schedule

        self.eps = eps

        self.projectors = nn.ModuleList()
        self.built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_once(self, s_feats, t_feats):
        """
        根据第一次 forward 得到的师生特征通道数构建 1x1 conv 映射：
            F_S <- conv1x1(F_S)

        注意：
            如果 student / teacher 通道数不同，则会创建 Conv2d。
            该 Conv2d 参数需要在训练循环里加入 optimizer。
        """
        if self.built:
            return False

        device = s_feats[0].device

        for i, (sf, tf) in enumerate(zip(s_feats, t_feats)):
            cs = sf.shape[1]
            ct = tf.shape[1]

            if cs == ct:
                projector = nn.Identity()
            else:
                projector = nn.Conv2d(
                    cs,
                    ct,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )

            projector = projector.to(device)
            self.projectors.append(projector)

            self.register_buffer(
                f"memory_{i}",
                torch.zeros(self.nc, ct, device=device),
            )
            self.register_buffer(
                f"valid_{i}",
                torch.zeros(self.nc, dtype=torch.bool, device=device),
            )

        self.built = True
        return True

    def _memory(self, layer_idx):
        return getattr(self, f"memory_{layer_idx}")

    def _valid(self, layer_idx):
        return getattr(self, f"valid_{layer_idx}")

    # ------------------------------------------------------------------
    # AKS
    # ------------------------------------------------------------------
    def compute_adaptation_score(self, r_lc, q_lc, sim_lc, g_lc, u_lc):
        """
        Compute A_l,c.

        Full:
            A_l,c = 0.24R + 0.24Q + 0.20Sim + 0.20G + 0.12(1-U)

        Ablation:
            通过 aks_use_xxx 控制是否启用某个组成项。
        """

        if not self.use_aks:
            return torch.ones_like(r_lc)

        terms = []
        weights = []

        if self.aks_use_r:
            terms.append(r_lc)
            weights.append(self.aks_w_r)

        if self.aks_use_q:
            terms.append(q_lc)
            weights.append(self.aks_w_q)

        if self.aks_use_sim:
            terms.append(sim_lc)
            weights.append(self.aks_w_sim)

        if self.aks_use_g:
            terms.append(g_lc)
            weights.append(self.aks_w_g)

        if self.aks_use_u:
            terms.append(1.0 - u_lc)
            weights.append(self.aks_w_u)

        if len(terms) == 0:
            return torch.ones_like(r_lc)

        score = torch.zeros_like(r_lc)

        for t, w in zip(terms, weights):
            score = score + w * t

        if self.aks_renorm_weights:
            weight_sum = max(sum(weights), self.eps)
            score = score / weight_sum

        score = score.clamp(0.0, 1.0)

        if self.aks_detach:
            score = score.detach()

        return score

    def get_review_lambda(self, epoch, epochs):
        """
        Review loss weight schedule.

        sine:
            lambda_re = sin(pi * e / E)
        """

        e = float(epoch + 1)
        E = max(float(epochs), 1.0)

        if self.review_lambda_schedule == "sine":
            return math.sin(math.pi * e / E)

        elif self.review_lambda_schedule == "constant":
            return 1.0

        elif self.review_lambda_schedule == "linear":
            return e / E

        elif self.review_lambda_schedule == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * e / E))

        elif self.review_lambda_schedule == "none":
            return 1.0

        else:
            return math.sin(math.pi * e / E)

    # ------------------------------------------------------------------
    # Feature / mask utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_norm(x, eps=1e-6):
        """
        Norm(F) = (F - mean) / std
        mean/std 在空间维度 H,W 上计算。
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (x - mean) / std

    def _make_fg_masks(self, batch, feats):
        """
        根据 GT 框构建 Ω_l,c^fg。

        默认 Ultralytics 格式：
            batch['bboxes']: normalized xywh, shape [N, 4]
            batch['cls']: shape [N, 1] or [N]
            batch['batch_idx']: shape [N]
        """

        b = feats[0].shape[0]
        device = feats[0].device

        masks = []
        for feat in feats:
            _, _, h, w = feat.shape
            masks.append(
                torch.zeros(
                    b,
                    self.nc,
                    h,
                    w,
                    device=device,
                    dtype=torch.bool,
                )
            )

        if "bboxes" not in batch or batch["bboxes"].numel() == 0:
            return masks

        batch_idx = batch["batch_idx"].long().view(-1)
        cls = batch["cls"].long().view(-1)
        boxes = batch["bboxes"]

        for bi, ci, box in zip(batch_idx, cls, boxes):
            bi = int(bi.item())
            ci = int(ci.item())

            if bi < 0 or bi >= b:
                continue

            if ci < 0 or ci >= self.nc:
                continue

            # normalized xywh
            cx, cy, bw, bh = box

            for li, feat in enumerate(feats):
                _, _, h, w = feat.shape

                x1 = ((cx - bw / 2) * w).floor().long().clamp(0, w - 1)
                y1 = ((cy - bh / 2) * h).floor().long().clamp(0, h - 1)
                x2 = ((cx + bw / 2) * w).ceil().long().clamp(0, w - 1)
                y2 = ((cy + bh / 2) * h).ceil().long().clamp(0, h - 1)

                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())

                if x2 < x1 or y2 < y1:
                    continue

                masks[li][bi, ci, y1:y2 + 1, x1:x2 + 1] = True

        return masks

    def _flatten_pred_tensors(self, pred):
        """
        从 teacher 输出中递归提取 Tensor。
        """
        tensors = []

        if torch.is_tensor(pred):
            tensors.append(pred)

        elif isinstance(pred, dict):
            for v in pred.values():
                tensors.extend(self._flatten_pred_tensors(v))

        elif isinstance(pred, (list, tuple)):
            for v in pred:
                tensors.extend(self._flatten_pred_tensors(v))

        return tensors

    def _extract_teacher_cls_maps(self, t_pred, t_feats):
        """
        尝试从 teacher raw pred 中提取每个检测层的类别概率图。

        对 YOLO 类模型：
            raw pred 可能是 list[Tensor]
            Tensor shape 可能是 [B, no, H, W]
            最后 nc 个通道通常是分类 logits。

        如果无法提取，则返回 None，后续 R=1, U=0 作为 fallback。
        """

        if t_pred is None:
            return None

        tensors = self._flatten_pred_tensors(t_pred)

        candidates = []
        for x in tensors:
            if torch.is_tensor(x) and x.ndim == 4 and x.shape[1] >= self.nc:
                candidates.append(x)

        if len(candidates) == 0:
            return None

        cls_maps = []

        for i, feat in enumerate(t_feats):
            raw = candidates[min(i, len(candidates) - 1)]

            cls_logits = raw[:, -self.nc:, :, :]
            prob = cls_logits.sigmoid()

            if prob.shape[-2:] != feat.shape[-2:]:
                prob = F.interpolate(
                    prob,
                    size=feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            cls_maps.append(prob.detach())

        return cls_maps

    def _calc_reliability_uncertainty(self, cls_map, mask, c):
        """
        R_l,c 和 U_l,c。

        R_l,c：
            前景区域类别 c 的平均教师置信度。

        U_l,c：
            前景区域 teacher 分类分布熵，归一化到 [0, 1]。
        """

        if cls_map is None or mask.sum().item() == 0:
            device = mask.device
            return (
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
            )

        p_c = cls_map[:, c, :, :][mask].clamp(self.eps, 1.0)
        r_lc = p_c.mean()

        all_p = cls_map.permute(0, 2, 3, 1)[mask].clamp(self.eps, 1.0)
        all_p = all_p / all_p.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        entropy = -(all_p * all_p.log()).sum(dim=-1)
        entropy = entropy / math.log(max(self.nc, 2))

        u_lc = entropy.mean().clamp(0.0, 1.0)

        return r_lc, u_lc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, s_feats, t_feats, batch, t_pred=None, epoch=0, epochs=1):
        """
        Return:
            proto_feature_loss:
                A_l,c 加权的类别原型特征蒸馏损失。

            review_loss:
                历史教师原型复习损失，已经乘上 lambda_re。

            adapt_mean:
                当前 batch 的平均 A_l,c，可用于重加权原始 CWD/MGD/KL feature loss。

            log_dict:
                用于 save_metrics() 的日志项。
        """

        self.build_once(s_feats, t_feats)

        s_feats = [proj(sf) for proj, sf in zip(self.projectors, s_feats)]
        t_feats = [tf.detach() for tf in t_feats]

        fg_masks = self._make_fg_masks(batch, t_feats)
        cls_maps = self._extract_teacher_cls_maps(t_pred, t_feats)

        proto_feature_loss = s_feats[0].new_zeros(())
        review_loss = s_feats[0].new_zeros(())

        adapt_values = []
        r_values = []
        q_values = []
        sim_values = []
        g_values = []
        u_values = []

        valid_pairs = 0
        memory_update_count = 0
        memory_valid_count = 0

        lambda_re = self.get_review_lambda(epoch, epochs)

        for li, (sf, tf, mask_l) in enumerate(zip(s_feats, t_feats, fg_masks)):
            sf_norm = self._spatial_norm(sf, self.eps)
            tf_norm = self._spatial_norm(tf, self.eps)

            memory = self._memory(li)
            valid = self._valid(li)

            memory_valid_count += int(valid.sum().item())

            cls_map = None if cls_maps is None else cls_maps[li]

            for c in range(self.nc):
                mask = mask_l[:, c, :, :]

                if mask.sum().item() == 0:
                    continue

                sf_points = sf.permute(0, 2, 3, 1)[mask]
                tf_points = tf.permute(0, 2, 3, 1)[mask]

                if sf_points.numel() == 0 or tf_points.numel() == 0:
                    continue

                # ------------------------------------------------------
                # P_S,l,c and P_T,l,c
                # ------------------------------------------------------
                ps = sf_points.mean(dim=0)
                pt = tf_points.mean(dim=0)

                # ------------------------------------------------------
                # Sim_l,c
                # ------------------------------------------------------
                sim_lc = 0.5 * (
                    1.0
                    + F.cosine_similarity(
                        ps.unsqueeze(0),
                        pt.unsqueeze(0),
                        dim=1,
                    ).squeeze(0)
                )
                sim_lc = sim_lc.clamp(0.0, 1.0)

                # ------------------------------------------------------
                # G_l,c
                # ------------------------------------------------------
                sf_norm_points = sf_norm.permute(0, 2, 3, 1)[mask]
                tf_norm_points = tf_norm.permute(0, 2, 3, 1)[mask]

                d_lc = torch.abs(sf_norm_points - tf_norm_points).mean()
                g_lc = torch.exp(-d_lc / self.aks_tau_g).clamp(0.0, 1.0)

                # ------------------------------------------------------
                # R_l,c and U_l,c
                # ------------------------------------------------------
                r_lc, u_lc = self._calc_reliability_uncertainty(
                    cls_map,
                    mask,
                    c,
                )

                # ------------------------------------------------------
                # Q_l,c
                # 当前未接入 teacher box decode，因此默认用 R_l,c 近似。
                # 如果你后续有 decoded teacher boxes，可以在这里替换为真实 IoU。
                # ------------------------------------------------------
                if self.use_q_fallback:
                    q_lc = r_lc
                else:
                    q_lc = torch.ones_like(r_lc)

                # ------------------------------------------------------
                # A_l,c
                # ------------------------------------------------------
                a_lc = self.compute_adaptation_score(
                    r_lc=r_lc,
                    q_lc=q_lc,
                    sim_lc=sim_lc,
                    g_lc=g_lc,
                    u_lc=u_lc,
                )

                adapt_values.append(a_lc.detach())
                r_values.append(r_lc.detach())
                q_values.append(q_lc.detach())
                sim_values.append(sim_lc.detach())
                g_values.append(g_lc.detach())
                u_values.append(u_lc.detach())

                # ------------------------------------------------------
                # 1. Prototype feature KD
                # ------------------------------------------------------
                if self.use_proto_feature_kd:
                    proto_feature_loss = proto_feature_loss + a_lc * F.mse_loss(ps, pt)

                # ------------------------------------------------------
                # 2. Historical prototype review
                # ------------------------------------------------------
                if self.use_review and valid[c]:
                    old_pt = memory[c].detach()

                    if self.review_use_consistency:
                        review_consistency = 0.5 * (
                            1.0
                            + F.cosine_similarity(
                                pt.unsqueeze(0),
                                old_pt.unsqueeze(0),
                                dim=1,
                            ).squeeze(0)
                        )
                        review_consistency = review_consistency.clamp(0.0, 1.0).detach()
                    else:
                        review_consistency = torch.ones_like(a_lc)

                    if self.review_use_adapt_gate:
                        review_gate = a_lc
                    else:
                        review_gate = torch.ones_like(a_lc)

                    review_loss = review_loss + review_gate * review_consistency * F.mse_loss(ps, old_pt)

                # ------------------------------------------------------
                # 3. Update historical teacher prototype memory
                # ------------------------------------------------------
                if self.use_review:
                    with torch.no_grad():
                        if self.review_update_with_aks:
                            allow_update = bool((a_lc.detach() > self.review_theta_h).item())
                        else:
                            allow_update = True

                        if allow_update:
                            if valid[c]:
                                memory[c].mul_(self.review_momentum).add_(
                                    pt.detach(),
                                    alpha=1.0 - self.review_momentum,
                                )
                            else:
                                memory[c].copy_(pt.detach())
                                valid[c] = True

                            memory_update_count += 1

                valid_pairs += 1

        if valid_pairs > 0:
            proto_feature_loss = proto_feature_loss / valid_pairs
            review_loss = review_loss / valid_pairs
            adapt_mean = torch.stack(adapt_values).mean()
        else:
            adapt_mean = s_feats[0].new_tensor(1.0)

        review_loss = review_loss * float(lambda_re)

        if len(adapt_values) > 0:
            a_stack = torch.stack(adapt_values)
            r_stack = torch.stack(r_values)
            q_stack = torch.stack(q_values)
            sim_stack = torch.stack(sim_values)
            g_stack = torch.stack(g_values)
            u_stack = torch.stack(u_values)

            log_dict = {
                "aks/A_mean": a_stack.mean(),
                "aks/A_min": a_stack.min(),
                "aks/A_max": a_stack.max(),
                "aks/R_mean": r_stack.mean(),
                "aks/Q_mean": q_stack.mean(),
                "aks/Sim_mean": sim_stack.mean(),
                "aks/G_mean": g_stack.mean(),
                "aks/U_mean": u_stack.mean(),
                "review/lambda": float(lambda_re),
                "review/memory_updates": float(memory_update_count),
                "review/memory_valid": float(memory_valid_count),
                "review/valid_pairs": float(valid_pairs),
            }
        else:
            log_dict = {
                "aks/A_mean": float(1.0),
                "aks/A_min": float(1.0),
                "aks/A_max": float(1.0),
                "aks/R_mean": float(1.0),
                "aks/Q_mean": float(1.0),
                "aks/Sim_mean": float(1.0),
                "aks/G_mean": float(1.0),
                "aks/U_mean": float(0.0),
                "review/lambda": float(lambda_re),
                "review/memory_updates": float(0.0),
                "review/memory_valid": float(memory_valid_count),
                "review/valid_pairs": float(0.0),
            }

        return proto_feature_loss, review_loss, adapt_mean, log_dict


def get_activation(feat, backbone_idx=-1):
    def hook(model, inputs, outputs):
        if backbone_idx != -1:
            for _ in range(5 - len(outputs)): outputs.insert(0, None)
            # for idx, i in enumerate(outputs):
            #     if i is None:
            #         print(idx, 'None')
            #     else:
            #         print(idx, i.size())
            feat.append(outputs[backbone_idx])
        else:
            feat.append(outputs)
    return hook

class DetectionDistiller(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.logical_disloss = None
        self.feature_disloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names) + 2)) % ('Epoch', 'GPU_mem', *self.loss_names, 'log_loss', 'fea_loss', 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5, feature_model=None):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            LOGGER.info(f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                        f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                        f"determining best 'optimizer', 'lr0' and 'momentum' automatically... ")
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if feature_model is not None:
            for v in feature_model.modules():
                for p_name, p in v.named_parameters(recurse=0):
                    if p_name == 'bias':  # bias (no decay)
                        g[2].append(p)
                    elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                        g[1].append(p)
                    else:
                        g[0].append(p)  # weight (with decay)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
    
    def setup_prune_model(self):
        ckpt = torch.load(self.model, map_location=self.device)
        model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(f"{colorstr('Loading Prune Student Model form {}'.format(self.model))}")
        self.model = model
        self.model.info()
        return ckpt
    
    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = weights.yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=self.pretrain_weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt
    
    def setup_teacher_model(self):
        """Load/create/download model for any task."""
        # model, weights = self.args.teacher_weights, None
        # ckpt = None
        # if str(model).endswith('.pt'):
        #     weights, ckpt = attempt_load_one_weight(model)
        #     cfg = weights.yaml
        # else:
        #     cfg = model
        # self.teacher_model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        
        ckpt = torch.load(self.args.teacher_weights, map_location=self.device)
        self.teacher_model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        self.teacher_model.train()
        self.teacher_model.info()
        return ckpt
    
    def _setup_train(self, world_size):
        # init model
        if self.args.prune_model:
            ckpt = self.setup_prune_model()
        else:
            LOGGER.info(f"{colorstr('SetUp Student Model:')}")
            ckpt = self.setup_model()
        LOGGER.info(f"{colorstr('SetUp Teacher Model:')}")
        _ = self.setup_teacher_model()
        
        self.model.to(self.device)
        self.teacher_model.to(self.device)
        self.set_model_attributes()
        self.model.criterion = self.model.init_criterion()
        
        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True
        
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Init Distill Loss
        self.kd_logical_loss, self.kd_feature_loss = None, None
        if self.args.kd_loss_type == 'logical' or self.args.kd_loss_type == 'all':
            self.kd_logical_loss = LogicalLoss(self.args, self.model, self.args.logical_loss_type)
        if self.args.kd_loss_type == 'feature' or self.args.kd_loss_type == 'all':
            s_feature, t_feature = [], []
            hooks = []
            self.teacher_kd_layers, self.student_kd_layers = self.args.teacher_kd_layers.split(','), self.args.student_kd_layers.split(',')
            s_feature_idx, t_feature_idx = [], []
            assert len(self.teacher_kd_layers) == len(self.student_kd_layers), f"teacher{self.teacher_kd_layers} and student{self.student_kd_layers} layers not equal.."
            for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                if '-' in t_layer:
                    t_layer_first, t_layer_second = t_layer.split('-')
                    t_feature_idx.append(int(t_layer_second) / 10)
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                else:
                    hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                    t_feature_idx.append(int(t_layer))
                
                if '-' in s_layer:
                    s_layer_first, s_layer_second = s_layer.split('-')
                    s_feature_idx.append(int(s_layer_second) / 10)
                    hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                else:
                    hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
                    s_feature_idx.append(int(s_layer))

            inputs = torch.randn((2, 3, self.args.imgsz, self.args.imgsz)).to(self.device)
            with torch.no_grad():
                _ = self.teacher_model(inputs)
                _ = self.model(inputs)
            s_feature_sort_idx, t_feature_sort_idx = sorted(s_feature_idx), sorted(t_feature_idx)
            s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
            t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]
            self.kd_feature_loss = FeatureLoss([s_feature[i].size(1) for i in s_feature_idx], [t_feature[i].size(1) for i in t_feature_idx], distiller=self.args.feature_loss_type)
            for hook in hooks:
                hook.remove()
        
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations,
                                              feature_model=self.kd_feature_loss if isinstance(self.kd_feature_loss, FeatureLoss) else None)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(None)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
    
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        
        ## add
        # ------------------------------------------------------------
        # Adaptive Knowledge Selection + Historical Prototype Review KD
        # ------------------------------------------------------------
        self.use_adaptive_review_kd = getattr(self.args, "use_adaptive_review_kd", True)

        if self.use_adaptive_review_kd:
            nc = getattr(de_parallel(self.model), "nc", None)

            if nc is None:
                nc = getattr(self, "data", {}).get("nc", None)

            if nc is None:
                nc = len(getattr(self, "names", []))

            self.adaptive_review_kd = AdaptiveReviewKD(
                nc=nc,

                # module-level ablation
                use_aks=getattr(self.args, "use_aks", True),
                use_review=getattr(self.args, "use_review", True),
                use_proto_feature_kd=getattr(self.args, "use_proto_feature_kd", True),

                # AKS component ablation
                aks_use_r=getattr(self.args, "aks_use_r", True),
                aks_use_q=getattr(self.args, "aks_use_q", True),
                aks_use_sim=getattr(self.args, "aks_use_sim", True),
                aks_use_g=getattr(self.args, "aks_use_g", True),
                aks_use_u=getattr(self.args, "aks_use_u", True),

                aks_w_r=getattr(self.args, "aks_w_r", 0.24),
                aks_w_q=getattr(self.args, "aks_w_q", 0.24),
                aks_w_sim=getattr(self.args, "aks_w_sim", 0.20),
                aks_w_g=getattr(self.args, "aks_w_g", 0.20),
                aks_w_u=getattr(self.args, "aks_w_u", 0.12),

                aks_tau_g=getattr(self.args, "aks_tau_g", 1.0),
                aks_detach=getattr(self.args, "aks_detach", True),
                aks_renorm_weights=getattr(self.args, "aks_renorm_weights", True),
                use_q_fallback=getattr(self.args, "use_q_fallback", True),

                # Review ablation
                review_use_adapt_gate=getattr(self.args, "review_use_adapt_gate", True),
                review_use_consistency=getattr(self.args, "review_use_consistency", True),
                review_update_with_aks=getattr(self.args, "review_update_with_aks", True),
                review_theta_h=getattr(self.args, "review_theta_h", 0.5),
                review_momentum=getattr(self.args, "review_momentum", 0.999),
                review_lambda_schedule=getattr(self.args, "review_lambda_schedule", "sine"),
            ).to(self.device)
        else:
            self.adaptive_review_kd = None

        self.aks_log_dict = {}
        self._adaptive_review_kd_params_added = False





        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            
            if self.args.kd_loss_type in ['feature', 'all']:
                self.kd_feature_loss.train()
                hooks = []
                s_feature, t_feature = [], []
                s_feature_idx, t_feature_idx = [], []
                for t_layer, s_layer in zip(self.teacher_kd_layers, self.student_kd_layers):
                    if '-' in t_layer:
                        t_layer_first, t_layer_second = t_layer.split('-')
                        t_feature_idx.append(int(t_layer_second) / 10)
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer_first)].register_forward_hook(get_activation(t_feature, backbone_idx=int(t_layer_second))))
                    else:
                        hooks.append(de_parallel(self.teacher_model).model[int(t_layer)].register_forward_hook(get_activation(t_feature)))
                        t_feature_idx.append(int(t_layer))
                    
                    if '-' in s_layer:
                        s_layer_first, s_layer_second = s_layer.split('-')
                        s_feature_idx.append(int(s_layer_second) / 10)
                        hooks.append(de_parallel(self.model).model[int(s_layer_first)].register_forward_hook(get_activation(s_feature, backbone_idx=int(s_layer_second))))
                    else:
                        hooks.append(de_parallel(self.model).model[int(s_layer)].register_forward_hook(get_activation(s_feature)))
                        s_feature_idx.append(int(s_layer))
                
                s_feature_sort_idx, t_feature_sort_idx = sorted(s_feature_idx), sorted(t_feature_idx)
                s_feature_idx = [s_feature_sort_idx.index(i) for i in s_feature_idx]
                t_feature_idx = [t_feature_sort_idx.index(i) for i in t_feature_idx]
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.logical_disloss = torch.zeros(1, device=self.device)
            self.feature_disloss = torch.zeros(1, device=self.device)
             ##add
            self.review_disloss = torch.zeros(1, device=self.device)
            self.proto_disloss = torch.zeros(1, device=self.device)

            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                if hasattr(de_parallel(self.model), 'net_update_temperature'):
                    temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                    de_parallel(self.model).net_update_temperature(temp)
                
                if self.args.kd_loss_decay == 'constant':
                    distill_decay = 1.0
                elif self.args.kd_loss_decay == 'cosine':
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear':
                    distill_decay = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.01 - 1) + 1
                elif self.args.kd_loss_decay == 'cosine_epoch':
                    eta_min, base_ratio, T_max = 0.01, 1.0, 10
                    distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * ni / T_max)) / 2
                elif self.args.kd_loss_decay == 'linear_epoch':
                    distill_decay = ((1 - math.cos(ni * math.pi / (self.epochs * nb))) / 2) * (0.01 - 1) + 1
                ## adapta  KD/FusionKD/mmdet/models/detectors/fusionkd_atss.py
                elif self.args.kd_loss_decay == 'adaptive':
                    progress = epoch / self.epochs
                    # 自适应衰减策略
                    if epoch < 10:
                        # 前期：强蒸馏，快速学习教师知识
                        distill_decay = 1.0 - progress * 0.2
                    elif epoch < 20:
                        # 中期：平衡蒸馏，防止过拟合
                        distill_decay = 0.8 - (progress - 0.33) * 0.4
                    else:
                        # 后期：弱蒸馏，主要优化检测任务
                        distill_decay = max(0.2, 0.4 - (progress - 0.66) * 0.5)
                elif self.args.kd_loss_decay == 'performance_aware':
                    # 性能感知衰减
                    if len(self.performance_history) > 3:
                        recent_improvement = self.performance_history[-1] - self.performance_history[-3]
                        if recent_improvement < 0.01:  # 性能提升缓慢
                            distill_decay = max(0.2, distill_decay * 0.8)
                        else:
                            distill_decay = min(1.0, distill_decay * 1.1)

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    pred = de_parallel(self.model).predict(batch['img'])
                    
                    with torch.no_grad():
                        t_pred = de_parallel(self.teacher_model).predict(batch['img'])
                    
                    main_loss, self.loss_items = de_parallel(self.model).criterion(pred, batch)
                    
                    log_distill_loss, fea_distill_loss = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
                    ##add
                    review_loss = torch.zeros(1, device=self.device)
                    proto_fea_loss = torch.zeros(1, device=self.device)
                    adapt_weight = torch.ones(1, device=self.device)
                    aks_log_dict = {}

                    if self.kd_logical_loss is not None:
                        if type(pred) is dict and type(t_pred) is dict:
                            log_distill_loss = self.kd_logical_loss(pred['one2one'], t_pred['one2one'], batch, True) * self.args.logical_loss_ratio
                            log_distill_loss += self.kd_logical_loss(pred['one2many'], t_pred['one2many'], batch) * self.args.logical_loss_ratio * 0.5
                        else:
                            log_distill_loss = self.kd_logical_loss(pred, t_pred, batch) * self.args.logical_loss_ratio
                    # if self.kd_feature_loss is not None:
                    #     fea_distill_loss = self.kd_feature_loss([s_feature[i] for i in s_feature_idx], [t_feature[i] for i in t_feature_idx]) * self.args.feature_loss_ratio
                    
                    ##add
                    # Feature KD + AKS + Review
                    if self.kd_feature_loss is not None:
                        s_kd_feats = [s_feature[i] for i in s_feature_idx]
                        t_kd_feats = [t_feature[i] for i in t_feature_idx]

                        # 原始公开可复现特征蒸馏损失，例如 CWD / MGD / KL
                        base_fea_distill_loss = self.kd_feature_loss(
                            s_kd_feats,
                            t_kd_feats,
                        )

                        if self.adaptive_review_kd is not None:
                            was_built = self.adaptive_review_kd.built

                            proto_fea_loss, review_loss, adapt_weight, aks_log_dict = self.adaptive_review_kd(
                                s_kd_feats,
                                t_kd_feats,
                                batch=batch,
                                t_pred=t_pred,
                                epoch=epoch,
                                epochs=self.epochs,
                            )

                            # 如果第一次 forward 创建了 1x1 Conv projector，需要加入 optimizer
                            if (not was_built) and (not self._adaptive_review_kd_params_added):
                                new_params = [
                                    p for p in self.adaptive_review_kd.projectors.parameters()
                                    if p.requires_grad
                                ]

                                if len(new_params) > 0:
                                    base_pg = self.optimizer.param_groups[0]

                                    self.optimizer.add_param_group({
                                        "params": new_params,
                                        "lr": base_pg["lr"],
                                        "initial_lr": base_pg.get("initial_lr", base_pg["lr"]),
                                        "momentum": base_pg.get("momentum", getattr(self.args, "momentum", 0.937)),
                                        "weight_decay": base_pg.get("weight_decay", 0.0),
                                    })

                                self._adaptive_review_kd_params_added = True

                            # 是否用 A_l,c 重加权原始 feature KD
                            if getattr(self.args, "use_aks", True):
                                base_fea_distill_loss = base_fea_distill_loss * adapt_weight

                            # 最终 feature KD：
                            #   原 feature KD + 类别原型 feature KD
                            fea_distill_loss = (
                                base_fea_distill_loss + proto_fea_loss
                            ) * self.args.feature_loss_ratio

                        else:
                            fea_distill_loss = base_fea_distill_loss * self.args.feature_loss_ratio

                    if aks_log_dict:
                        self.aks_log_dict = aks_log_dict


                    # self.loss = main_loss + (
                    #     log_distill_loss + fea_distill_loss
                    #  ) * batch['img'].size(0) * distill_decay
                    
                    review_loss_ratio = getattr(self.args, "review_loss_ratio", 0.5)
                    
                    # L = L_det + L_logical + L_feature + λ_review * L_review
                    # L_feature = A_l,c 加权后的原 feature KD + A_l,c 加权的类别原型 KD
                    # L_review = λ_re * A_l,c * r_l,c * ||P_S,l,c - M_l,c||^2
                    self.loss = main_loss + (
                        log_distill_loss
                        + fea_distill_loss
                        + review_loss * review_loss_ratio
                    ) * batch["img"].size(0) * distill_decay


                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items
                    self.logical_disloss = (self.logical_disloss * i + log_distill_loss) / (i + 1) if self.logical_disloss is not None \
                        else log_distill_loss
                    self.feature_disloss = (self.feature_disloss * i + fea_distill_loss) / (i + 1) if self.feature_disloss is not None \
                        else fea_distill_loss
                    ##add,滑动统计
                    self.review_disloss = (self.review_disloss * i + review_loss) / (i + 1) if self.review_disloss is not None \
                        else review_loss
                    self.proto_disloss = (self.proto_disloss * i + proto_fea_loss) / (i + 1) if self.proto_disloss is not None \
                        else proto_fea_loss
                    
                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                logical_dislosses = self.logical_disloss if loss_len > 1 else torch.unsqueeze(self.logical_disloss, 0)
                feature_dislosses = self.feature_disloss if loss_len > 1 else torch.unsqueeze(self.feature_disloss, 0)
                ##add,更新进度条
                review_dislosses = self.review_disloss if loss_len > 1 else torch.unsqueeze(self.review_disloss, 0)
                proto_dislosses = self.proto_disloss if loss_len > 1 else torch.unsqueeze(self.proto_disloss, 0)


                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len + 2)) %
                        (f'{epoch + 1}/{self.epochs}',
                          mem, 
                          *losses,
                          *logical_dislosses, 
                          *feature_dislosses,
                        *proto_dislosses,
                        *review_dislosses,
                            batch['cls'].shape[0], 
                            batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')
                
                if self.kd_feature_loss is not None:
                    s_feature.clear()
                    t_feature.clear()

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')
            
            if self.kd_feature_loss is not None:
                for hook in hooks:
                    hook.remove()

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                # self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                
                extra_kd_metrics = {}
                if hasattr(self, "aks_log_dict") and self.aks_log_dict:
                    for k, v in self.aks_log_dict.items():
                        if torch.is_tensor(v):
                            extra_kd_metrics[k] = float(v.detach().cpu())
                        else:
                            extra_kd_metrics[k] = float(v)
                extra_kd_metrics["kd/logical_loss"] = float(self.logical_disloss.detach().cpu().mean())
                extra_kd_metrics["kd/feature_loss"] = float(self.feature_disloss.detach().cpu().mean())
                extra_kd_metrics["kd/proto_loss"] = float(self.proto_disloss.detach().cpu().mean())
                extra_kd_metrics["kd/review_loss"] = float(self.review_disloss.detach().cpu().mean())

                self.save_metrics(
                    metrics={
                        **self.label_loss_items(self.tloss),
                        **self.metrics,
                        **self.lr,
                        **extra_kd_metrics,
                    }
                )


                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
        
    def distill(self, weights=None):
        self.pretrain_weights = weights
        self.train()