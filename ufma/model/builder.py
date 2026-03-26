# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import warnings

import nncore
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

ATTN_IMPLEMENTATION = 'sdpa'
PARTIAL_WEIGHT_NAME = 'pytorch_model.safetensors'
FULL_MODEL_CLASSES = {'qwen2_vl': Qwen2VLForConditionalGeneration}

if Qwen2_5_VLForConditionalGeneration is not None:
    FULL_MODEL_CLASSES['qwen2_5_vl'] = Qwen2_5_VLForConditionalGeneration


def get_auto_device():
    try:
        import torch_npu
        has_npu = torch_npu.npu.is_available()
    except ImportError:
        has_npu = False

    if torch.cuda.is_available():
        return 'cuda'
    if has_npu:
        return 'npu'
    return 'cpu'


def _resolve_device_map(device):
    return 'auto' if device == 'all' else None


def _load_processor(model_path):
    # Let the repo's video preprocessing control resizing instead of the processor.
    return AutoProcessor.from_pretrained(model_path, do_resize=False)


def _get_partial_paths(model_path, config):
    adapter_path = nncore.join(model_path, getattr(config, 'role', 'unknown'))
    partial_path = nncore.join(model_path, PARTIAL_WEIGHT_NAME)
    return adapter_path, partial_path


def _needs_partial_loading(adapter_path, partial_path):
    return nncore.is_dir(adapter_path) or nncore.is_file(partial_path)


def _load_generation_config(model, model_path):
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except OSError:
        warnings.warn('generation_config.json not found')


def _materialize_meta_parameters(model):
    meta_state_dict = {
        name: torch.empty_like(param, device='cpu')
        for name, param in model.named_parameters() if param.device == torch.device('meta')
    }
    if meta_state_dict:
        model.load_state_dict(meta_state_dict, strict=False, assign=True)


def _resize_projection_if_needed(weight, expected_shape, label):
    if weight.size() == expected_shape:
        return weight

    print(f'Resizing {label} to {expected_shape}...')
    return nn.Parameter(weight.new_empty(expected_shape))


def _ensure_token_layers_initialized(model):
    embed_shape = (model.model.embed_tokens.num_embeddings, model.model.embed_tokens.embedding_dim)
    model.model.embed_tokens.weight = _resize_projection_if_needed(model.model.embed_tokens.weight, embed_shape,
                                                                   'embed_tokens')

    lm_head_shape = (model.lm_head.out_features, model.lm_head.in_features)
    model.lm_head.weight = _resize_projection_if_needed(model.lm_head.weight, lm_head_shape, 'lm_head')


def _load_partial_artifacts(model_path, config, adapter_path, partial_path, is_trainable, merge_adapter, dtype,
                            device):
    print(f'Loading base model from {config.base_model_path}...')
    model = AutoModel.from_pretrained(
        config.base_model_path,
        config=config,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=dtype,
        device_map=_resolve_device_map(device))

    _load_generation_config(model, model_path)
    _materialize_meta_parameters(model)
    _ensure_token_layers_initialized(model)

    if nncore.is_dir(adapter_path):
        print(f'Loading adapter from {adapter_path}...')
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name=config.role,
            is_trainable=is_trainable,
            low_cpu_mem_usage=True,
            torch_device=str(model.model.embed_tokens.weight.device))

    if nncore.is_file(partial_path):
        print(f'Loading state dict from {partial_path}...')
        _, unexpected = load_model(model, partial_path, strict=False, device=str(model.device))
        assert len(unexpected) == 0, f'unexpected parameters: {unexpected}'

    if merge_adapter and nncore.is_dir(adapter_path):
        print('Merging adapter and unloading...')
        model = model.merge_and_unload()
        model._hf_peft_config_loaded = False

    return model


def _load_full_model(model_path, config, dtype, device):
    print(f'Loading full model from {model_path}...')

    if len(config.architectures) == 1 and config.model_type in FULL_MODEL_CLASSES:
        model_cls = FULL_MODEL_CLASSES[config.model_type]
    elif config.model_type == 'qwen2_5_vl' and Qwen2_5_VLForConditionalGeneration is None:
        raise ImportError('Qwen2.5-VL support requires transformers>=4.50.0 with qwen2_5_vl available')
    else:
        model_cls = AutoModel

    return model_cls.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=dtype,
        device_map=_resolve_device_map(device))


def build_model(model_path, config=None, is_trainable=False, merge_adapter=False, device='auto', dtype=torch.float16):
    processor = _load_processor(model_path)
    config = config or AutoConfig.from_pretrained(model_path)

    adapter_path, partial_path = _get_partial_paths(model_path, config)
    if _needs_partial_loading(adapter_path, partial_path):
        model = _load_partial_artifacts(model_path, config, adapter_path, partial_path, is_trainable, merge_adapter,
                                        dtype, device)
    else:
        model = _load_full_model(model_path, config, dtype, device)

    if not is_trainable and device != 'all':
        runtime_device = get_auto_device() if device == 'auto' else device
        model = model.to(runtime_device).eval()

    return model, processor
