# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import warnings

import nncore
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import ModuleList, PositionalEncoding, Sequential, TransformerEncoderLayer, xavier_init_
from nncore.ops import temporal_iou
from transformers import AutoConfig, AutoModel, Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLModel
from transformers.activations import ACT2CLS, ACT2FN
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from .blocks import ConvHead, ConvPyramid, LearnableEmbedding, Scale
from .generator import PointGenerator
from .loss import BundleLoss

try:
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    HAS_QWEN2_5_VL = True
except ImportError:
    Qwen2_5_VLConfig = None
    Qwen2_5_VLForConditionalGeneration = None
    Qwen2_5_VLModel = None
    Qwen2_5_VisionTransformerPretrainedModel = None
    HAS_QWEN2_5_VL = False


def cache_state_hook(module, args):
    module.state = args[0]


class AgentQwen2VLConfig(Qwen2VLConfig):
    model_type = 'agent_qwen2_vl'


if HAS_QWEN2_5_VL:

    class AgentQwen2_5_VLConfig(Qwen2_5_VLConfig):
        model_type = 'agent_qwen2_5_vl'


class _AgentVisionTransformerMixin:

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False

    # add support for gradient checkpointing
    # https://github.com/huggingface/transformers/pull/34724
    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(blk.__call__, hidden_states, cu_seqlens,
                                                                  rotary_pos_emb)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


class AgentQwen2VisionTransformerPretrainedModel(_AgentVisionTransformerMixin, Qwen2VisionTransformerPretrainedModel):
    pass


if HAS_QWEN2_5_VL:

    class AgentQwen2_5_VisionTransformerPretrainedModel(_AgentVisionTransformerMixin,
                                                        Qwen2_5_VisionTransformerPretrainedModel):
        pass


class _AgentBackboneMixin:

    def __init__(self, config):
        super().__init__(config)
        self.norm.register_forward_pre_hook(cache_state_hook)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # ensure gradient tracking (in case that embed_tokens has been frozen)
        assert input_ids is None and inputs_embeds is not None
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


class AgentQwen2VLModel(_AgentBackboneMixin, Qwen2VLModel):
    config_class = AgentQwen2VLConfig


if HAS_QWEN2_5_VL:

    class AgentQwen2_5_VLModel(_AgentBackboneMixin, Qwen2_5_VLModel):
        config_class = AgentQwen2_5_VLConfig


class _AgentVisionLanguageMixin:
    vision_model_cls = None
    backbone_model_cls = None

    def __init__(self, config):
        super().__init__(config)
        self.visual = self.vision_model_cls._from_config(config.vision_config)
        self.model = self.backbone_model_cls(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None

        if self.config.role in ('all_in_one', 'grounder'):
            self._init_grounding_modules()

        self.post_init()

    def _init_grounding_modules(self):
        hidden_size, hidden_act = self.config.hidden_size, self.config.hidden_act

        self.dims = 256
        self.vis_proj = Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.dims))
        self.reg_proj = Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.dims))
        self.vis_norm = nn.LayerNorm(self.dims)
        self.vis_fuse = ModuleList(
            TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]),
            TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]),
            TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]))

        self.vis_pos = PositionalEncoding(self.dims, normalize=True, learnable=False)
        self.vis_emb = LearnableEmbedding(self.dims)
        self.reg_emb = LearnableEmbedding(self.dims)

        self.strides = (1, 2, 4, 8)
        self.vis_pad_length = self.strides[-1]

        self.pyramid = ConvPyramid(self.dims, self.strides, act_cls=ACT2CLS[hidden_act])
        self.class_head = ConvHead(self.dims, 1, act_cls=ACT2CLS[hidden_act])
        self.coord_head = ConvHead(self.dims, 2, act_cls=ACT2CLS[hidden_act])

        self.generator = PointGenerator(self.strides, 1024)
        self.coef = Scale(self.strides)
        self.bundle_loss = BundleLoss(
            sample_radius=1.5,
            loss_cls=dict(type='FocalLoss', reduction='none', loss_weight=5.0),
            loss_reg=dict(type='L1Loss', reduction='none', loss_weight=1.0),
            loss_sal=dict(type='SampledNCELoss', direction='row', loss_weight=0.05))

    def reset_conv_parameters(self):
        for block_name in ('pyramid', 'class_head', 'coord_head'):
            block = getattr(self, block_name, None)
            if block is None:
                continue
            for name, module in block.named_modules():
                if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                    print(f'Reset parameters of {block.__class__.__name__} {name} ({module.__class__.__name__})')
                    xavier_init_(module, distribution='uniform')

    def _cache_vision_ranges(self, input_ids):
        vision_start_inds = torch.nonzero(input_ids == self.config.vision_start_token_id).tolist()
        vision_end_inds = torch.nonzero(input_ids == self.config.vision_end_token_id).tolist()
        assert len(vision_start_inds) == len(vision_end_inds)

        cached_ranges = [[] for _ in range(input_ids.size(0))]
        for start_ind, end_ind in zip(vision_start_inds, vision_end_inds):
            assert start_ind[0] == end_ind[0]
            cached_ranges[start_ind[0]].append([start_ind[1] + 1, end_ind[1]])
        return cached_ranges

    def _get_video_window(self, video_grid_thw):
        return int(video_grid_thw[0][1] * video_grid_thw[0][2] / 4)

    def _pool_video_tokens(self, hidden_states, start, end, window):
        video_tokens = hidden_states[:, start:end]
        video_tokens = video_tokens.transpose(-1, -2)
        video_tokens = F.avg_pool1d(video_tokens.float(), window, stride=window).to(video_tokens.dtype)
        return video_tokens.transpose(-1, -2)

    def _fuse_joint_tokens(self, joint_tokens, collect_all_layers=False):
        if collect_all_layers:
            fused_tokens = []
            for block in self.vis_fuse:
                joint_tokens = block(joint_tokens)
                fused_tokens.append(joint_tokens)
            joint_tokens = torch.cat(fused_tokens)
        else:
            for block in self.vis_fuse:
                joint_tokens = block(joint_tokens)
        return self.vis_norm(joint_tokens)

    def _build_temporal_context(self, hidden_states, reg_tokens, vision_range, video_grid_thw, collect_all_layers=False):
        start, end = vision_range
        window = self._get_video_window(video_grid_thw)
        assert video_grid_thw[0][0] * window == end - start

        video_tokens = self._pool_video_tokens(hidden_states, start, end, window)
        video_tokens = self.vis_proj(video_tokens).repeat(reg_tokens.size(0), 1, 1)

        video_tokens = self.vis_emb(video_tokens)
        reg_tokens = self.reg_emb(reg_tokens)
        position_encoding = self.vis_pos(video_tokens).to(video_tokens.dtype)

        joint_tokens = torch.cat((video_tokens + position_encoding, reg_tokens), dim=1)
        joint_tokens = self._fuse_joint_tokens(joint_tokens, collect_all_layers=collect_all_layers)

        video_emb = joint_tokens[:, :-1]
        query_emb = joint_tokens[:, -1:]
        video_mask = video_emb.new_ones(video_emb.size(0), video_emb.size(1))
        return video_emb, query_emb, video_mask

    def _maybe_pad_video_embeddings(self, video_emb, video_mask):
        if video_emb.size(1) >= self.vis_pad_length:
            return video_emb, video_mask

        batch, num_frames, channels = video_emb.size()
        emb_padding = video_emb.new_zeros(batch, self.vis_pad_length - num_frames, channels)
        mask_padding = video_mask.new_zeros(batch, self.vis_pad_length - num_frames)
        return torch.cat((video_emb, emb_padding), dim=1), torch.cat((video_mask, mask_padding), dim=1)

    def _run_grounding_heads(self, video_emb, video_mask, return_mask=False, apply_sigmoid=False):
        padded_emb, padded_mask = self._maybe_pad_video_embeddings(video_emb, video_mask)

        if return_mask:
            pyramid_feats, pyramid_mask = self.pyramid(padded_emb, padded_mask, return_mask=True)
        else:
            pyramid_feats = self.pyramid(video_emb, video_mask)
            pyramid_mask = None

        points = self.generator(pyramid_feats)
        class_logits = [self.class_head(feat) for feat in pyramid_feats]
        if apply_sigmoid:
            class_logits = [logits.sigmoid() for logits in class_logits]
        out_class = torch.cat(class_logits, dim=1)

        out_coord = [self.coef(self.coord_head(feat).exp(), idx) for idx, feat in enumerate(pyramid_feats)]
        out_coord = torch.cat(out_coord, dim=1)

        return dict(point=points, out_class=out_class, out_coord=out_coord, pyramid_mask=pyramid_mask)

    def _decode_boundaries(self, point, out_coord, out_class, num_frames):
        saliency = out_class[0]
        boundary = out_coord[0]

        boundary[:, 0] *= -1
        boundary *= point[:, 3, None].repeat(1, 2)
        boundary += point[:, 0, None].repeat(1, 2)
        boundary /= num_frames
        return torch.cat((boundary, saliency), dim=-1), saliency

    def _apply_temporal_nms(self, boundary, nms_type='normal', threshold=0.75, sigma=0.5):
        assert nms_type in ('normal', 'linear', 'gaussian')

        _, sorted_inds = boundary[:, -1].sort(descending=True)
        boundary = boundary[sorted_inds]

        for idx in range(boundary.size(0)):
            max_idx = boundary[idx:, -1].argmax(dim=0)
            boundary = nncore.swap_element(boundary, idx, max_idx + idx)
            iou = temporal_iou(boundary[idx, None, :-1], boundary[idx + 1:, :-1])[0]

            if nms_type == 'normal':
                boundary[idx + 1:, -1][iou >= threshold] = 0
            elif nms_type == 'linear':
                boundary[idx + 1:, -1] *= 1 - iou
            else:
                boundary[idx + 1:, -1] *= (-iou.pow(2) / sigma).exp()

        return boundary

    def _select_regression_loss(self, losses, avg_factors):
        assert len(losses) in (1, 2) and len(losses) == len(avg_factors)
        if len(losses) == 2 and losses[0] > losses[1]:
            return losses[1], avg_factors[1]
        return losses[0], avg_factors[0]

    def _is_prefill_stage(self, past_key_values):
        if past_key_values is None:
            return True
        if hasattr(past_key_values, 'get_seq_length'):
            return past_key_values.get_seq_length() == 0
        try:
            return len(past_key_values) == 0
        except TypeError:
            return False

    def _prepare_position_inputs(self,
                                 input_ids,
                                 attention_mask,
                                 position_ids,
                                 image_grid_thw,
                                 video_grid_thw,
                                 rope_deltas,
                                 cache_position=None,
                                 second_per_grid_ts=None):
        if position_ids is None and input_ids is not None:
            position_ids, computed_rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw,
                                                                     attention_mask)
            if rope_deltas is None:
                rope_deltas = computed_rope_deltas
        return position_ids, rope_deltas, dict()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                rope_deltas=None,
                timestamps=None,
                saliency=None,
                pos_clip=None,
                cache_position=None,
                second_per_grid_ts=None,
                **kwargs):
        mode = 'training' if self.training else 'caching' if self._is_prefill_stage(past_key_values) else 'generating'

        position_ids, rope_deltas, extra_forward_kwargs = self._prepare_position_inputs(
            input_ids,
            attention_mask,
            position_ids,
            image_grid_thw,
            video_grid_thw,
            rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts)

        if mode in ('training', 'caching'):
            self.cache_vision_inds = self._cache_vision_ranges(input_ids)

        # move input ids to the correct device
        input_ids = input_ids.to(self.model.embed_tokens.weight.device)

        forward_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=not self.training if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas)
        forward_kwargs.update(extra_forward_kwargs)
        forward_kwargs.update(kwargs)

        outputs = super().forward(**forward_kwargs)

        if mode == 'caching':
            self.cache_norm_state = self.model.norm.state
            self.reg = []
            self.sal = []

        if mode == 'training' and timestamps is not None:
            loss_regs, avg_factors = [], []
            shift_labels = labels[..., 1:].contiguous()
            for batch_idx, (vision_inds, ts) in enumerate(zip(self.cache_vision_inds, timestamps)):
                vision_range = vision_inds[0]
                inds = torch.where(shift_labels[batch_idx] == self.config.reg_token_id)[0]
                reg_tokens = self.reg_proj(self.model.norm.state[batch_idx, inds, None])

                video_emb, query_emb, video_msk = self._build_temporal_context(
                    self.model.norm.state[batch_idx, None],
                    reg_tokens,
                    vision_range,
                    video_grid_thw,
                    collect_all_layers=True)

                grounding_outputs = self._run_grounding_heads(video_emb, video_msk, return_mask=True)
                pymid_msk = grounding_outputs['pyramid_mask']
                point = grounding_outputs['point']
                out_class = grounding_outputs['out_class']
                out_coord = grounding_outputs['out_coord']

                if len(pymid_msk) != len(self.strides):
                    warnings.warn(f'pyramid size mismatch: {len(pymid_msk)} {len(self.strides)}')

                data = dict(
                    point=point,
                    video_emb=video_emb,
                    query_emb=query_emb,
                    video_msk=video_msk,
                    pymid_msk=pymid_msk,
                    out_class=out_class,
                    out_coord=out_coord,
                    boundary=point.new_tensor(ts),
                    saliency=saliency[batch_idx].unsqueeze(0),
                    pos_clip=pos_clip[batch_idx].unsqueeze(0))

                losses = self.bundle_loss(data, dict())
                loss_regs.append(sum(v for v in losses.values()))
                avg_factors.append(len(ts))

            loss_reg, avg_factor = self._select_regression_loss(loss_regs, avg_factors)
            if avg_factor > 0:
                outputs.loss = outputs.loss + loss_reg / avg_factor
        elif mode == 'generating':
            logits = outputs.logits[0, -1]
            if logits.argmax() == self.config.reg_token_id:
                assert self.model.norm.state.size() == (1, 1, self.config.hidden_size)

                vision_range = self.cache_vision_inds[0][0]
                reg_tokens = self.reg_proj(self.model.norm.state)

                video_emb, _, video_msk = self._build_temporal_context(
                    self.cache_norm_state,
                    reg_tokens,
                    vision_range,
                    video_grid_thw,
                    collect_all_layers=False)

                num_frames = video_emb.size(1)
                grounding_outputs = self._run_grounding_heads(video_emb, video_msk, apply_sigmoid=True)
                bnd, sal = self._decode_boundaries(
                    grounding_outputs['point'],
                    grounding_outputs['out_coord'],
                    grounding_outputs['out_class'],
                    num_frames)
                bnd = self._apply_temporal_nms(bnd)

                self.reg.append(bnd[:100])
                self.sal.append(sal)

        return outputs


class AgentQwen2VLForConditionalGeneration(_AgentVisionLanguageMixin, Qwen2VLForConditionalGeneration):
    config_class = AgentQwen2VLConfig
    vision_model_cls = AgentQwen2VisionTransformerPretrainedModel
    backbone_model_cls = AgentQwen2VLModel


if HAS_QWEN2_5_VL:

    class AgentQwen2_5_VLForConditionalGeneration(_AgentVisionLanguageMixin, Qwen2_5_VLForConditionalGeneration):
        config_class = AgentQwen2_5_VLConfig
        vision_model_cls = AgentQwen2_5_VisionTransformerPretrainedModel
        backbone_model_cls = AgentQwen2_5_VLModel

        def _prepare_position_inputs(self,
                                     input_ids,
                                     attention_mask,
                                     position_ids,
                                     image_grid_thw,
                                     video_grid_thw,
                                     rope_deltas,
                                     cache_position=None,
                                     second_per_grid_ts=None):
            extra_forward_kwargs = dict()
            if cache_position is not None:
                extra_forward_kwargs['cache_position'] = cache_position
            if second_per_grid_ts is not None:
                extra_forward_kwargs['second_per_grid_ts'] = second_per_grid_ts
            return position_ids, rope_deltas, extra_forward_kwargs


MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[AgentQwen2VLConfig.model_type] = 'AgentQwen2VLForConditionalGeneration'

AutoConfig.register(AgentQwen2VLConfig.model_type, AgentQwen2VLConfig)
AutoModel.register(AgentQwen2VLConfig, AgentQwen2VLForConditionalGeneration)

if HAS_QWEN2_5_VL:
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[AgentQwen2_5_VLConfig.model_type] = 'AgentQwen2_5_VLForConditionalGeneration'
    AutoConfig.register(AgentQwen2_5_VLConfig.model_type, AgentQwen2_5_VLConfig)
    AutoModel.register(AgentQwen2_5_VLConfig, AgentQwen2_5_VLForConditionalGeneration)
