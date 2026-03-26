# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import LOSSES, Parameter, build_loss


@LOSSES.register()
class SampledNCELoss(nn.Module):

    def __init__(self, temperature=0.07, max_scale=100, learnable=False, direction=('row', 'col'), loss_weight=1.0):
        super().__init__()

        scale = torch.Tensor([math.log(1 / temperature)])
        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction,) if isinstance(direction, str) else direction
        self.loss_weight = loss_weight

    def _build_loss_mask(self, saliency, video_mask, pos_clip):
        batch_indices = torch.arange(saliency.size(0), device=saliency.device)
        pos_scores = saliency[batch_indices, pos_clip].unsqueeze(-1)
        loss_mask = (saliency <= pos_scores) * video_mask

        if not loss_mask.any():
            warnings.warn(f'loss_msk is all zeros: {loss_mask} {saliency} {video_mask} {pos_clip}')
        return loss_mask

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_indices = torch.arange(video_emb.size(0), device=video_emb.device)
        loss_mask = self._build_loss_mask(saliency, video_msk, pos_clip)

        scale = self.scale.exp().clamp(max=self.max_scale)
        similarity = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        similarity = similarity + torch.where(loss_mask > 0, 0.0, float('-inf'))

        loss = 0
        if 'row' in self.direction:
            row_log_prob = F.log_softmax(similarity, dim=1)[batch_indices, pos_clip]
            loss = loss - row_log_prob.sum()

        if 'col' in self.direction:
            col_log_prob = F.log_softmax(similarity.t(), dim=1)[pos_clip, batch_indices]
            loss = loss - col_log_prob.sum() / col_log_prob.size(0)

        return loss * self.loss_weight


@LOSSES.register()
class BundleLoss(nn.Module):

    def __init__(self, sample_radius=1.5, loss_cls=None, loss_reg=None, loss_sal=None):
        super().__init__()
        self._loss_cls = build_loss(loss_cls)
        self._loss_reg = build_loss(loss_reg)
        self._loss_sal = build_loss(loss_sal)
        self.sample_radius = sample_radius

    def _repeat_to_match(self, target, source):
        repeat_factor = source.size(0) // target.size(0)
        if target.dim() == 2:
            return target.repeat(repeat_factor, 1)
        return target.repeat(repeat_factor, 1, 1)

    def _build_single_target(self, points, gt_boundary, gt_class):
        num_points, num_gt = points.size(0), gt_boundary.size(0)

        lengths = (gt_boundary[:, 1] - gt_boundary[:, 0])[None, :].repeat(num_points, 1)
        gt_segments = gt_boundary[None].expand(num_points, num_gt, 2)

        left = points[:, 0, None] - gt_segments[:, :, 0]
        right = gt_segments[:, :, 1] - points[:, 0, None]
        reg_target = torch.stack((left, right), dim=-1)

        if self.sample_radius > 0:
            centers = (gt_segments[:, :, 0] + gt_segments[:, :, 1]) / 2
            min_center = centers - points[:, 3, None] * self.sample_radius
            max_center = centers + points[:, 3, None] * self.sample_radius
            center_left = points[:, 0, None] - torch.maximum(min_center, gt_segments[:, :, 0])
            center_right = torch.minimum(max_center, gt_segments[:, :, 1]) - points[:, 0, None]
            cls_mask = torch.stack((center_left, center_right), dim=-1).min(-1)[0] >= 0
        else:
            cls_mask = reg_target.min(-1)[0] >= 0

        reg_distance = reg_target.max(-1)[0]
        reg_mask = torch.logical_and(reg_distance >= points[:, 1, None], reg_distance <= points[:, 2, None])

        lengths.masked_fill_(~cls_mask, float('inf'))
        lengths.masked_fill_(~reg_mask, float('inf'))

        min_lengths, min_indices = lengths.min(dim=1)
        min_mask = torch.logical_and(lengths <= (min_lengths[:, None] + 1e-3), lengths < float('inf')).to(
            reg_target.dtype)

        labels = F.one_hot(gt_class[:, 0], 2).to(reg_target.dtype)
        cls_target = torch.matmul(min_mask, labels).clamp(min=0.0, max=1.0)[:, 1]
        reg_target = reg_target[range(num_points), min_indices] / points[:, 3, None]
        return cls_target, reg_target

    def _build_targets(self, data):
        cls_targets = []
        reg_targets = []

        for batch_idx in range(data['boundary'].size(0)):
            gt_boundary = data['boundary'][batch_idx] * data['video_emb'].size(1)
            gt_class = gt_boundary.new_ones(gt_boundary.size(0), 1).long()

            cls_target, reg_target = self._build_single_target(data['point'], gt_boundary, gt_class)
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)

        return torch.stack(cls_targets), torch.stack(reg_targets)

    def _compute_cls_loss(self, data, output, cls_target):
        logits = data['out_class'].squeeze(-1)
        valid_mask = torch.cat(data['pymid_msk'], dim=1)
        cls_target = self._repeat_to_match(cls_target, logits)

        loss_cls = self._loss_cls(logits, cls_target, weight=valid_mask)
        output['loss_cls'] = (loss_cls.sum(dim=1) / valid_mask.sum(dim=1)).sum()
        return output

    def _compute_reg_loss(self, data, output, cls_target, reg_target):
        coords = data['out_coord']
        reg_target = self._repeat_to_match(reg_target, coords)

        reg_mask = cls_target.unsqueeze(2).repeat(1, 1, 2).bool()
        reg_mask = self._repeat_to_match(reg_mask, coords)
        assert reg_mask.any(), 'empty mask in reg loss'

        loss_reg = self._loss_reg(coords, reg_target, weight=reg_mask)
        output['loss_reg'] = (loss_reg.sum(dim=[1, 2]) / reg_mask.sum(dim=[1, 2])).sum()
        return output

    def _compute_saliency_loss(self, data, output):
        video_emb = data['video_emb']
        query_emb = data['query_emb']
        video_mask = data['video_msk']
        saliency = self._repeat_to_match(data['saliency'], video_emb)
        pos_clip = data['pos_clip'][:, 0].repeat(video_emb.size(0) // data['pos_clip'].size(0))

        output['loss_sal'] = self._loss_sal(video_emb, query_emb, video_mask, saliency, pos_clip)
        return output

    def forward(self, data, output):
        if self._loss_reg is not None:
            cls_target, reg_target = self._build_targets(data)
            output = self._compute_reg_loss(data, output, cls_target, reg_target)
        else:
            cls_target = data['saliency']

        if self._loss_cls is not None:
            output = self._compute_cls_loss(data, output, cls_target)

        if self._loss_sal is not None:
            output = self._compute_saliency_loss(data, output)

        return output
