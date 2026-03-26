# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

from dataclasses import dataclass, field
from typing import Optional

import nncore
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments as HFTrainingArguments

from ufma.constants import REG_TOKEN, SEG_E_TOKEN, SEG_S_TOKEN
from ufma.dataset import HybridDataCollator, HybridDataset
from ufma.model import MODELS
from ufma.model.builder import build_model
from ufma.train.custom_trainer import CustomTrainer

HEAD_KEYS = [
    'vis_proj', 'reg_proj', 'vis_fuse', 'vis_norm', 'vis_pos', 'vis_emb', 'reg_emb', 'pyramid', 'class_head',
    'coord_head', 'coef', 'bundle_loss'
]
SUPPORTED_BASE_MODELS = ('qwen2_vl', 'qwen2_5_vl')
AGENT_MODEL_TYPES = ('agent_qwen2_vl', 'agent_qwen2_5_vl')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default=None)
    role: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    datasets: Optional[str] = field(default=None)
    min_video_len: Optional[int] = field(default=-1)
    max_video_len: Optional[int] = field(default=-1)
    min_num_words: Optional[int] = field(default=-1)
    max_num_words: Optional[int] = field(default=-1)
    max_retries: Optional[int] = field(default=10)


@dataclass
class CustomArguments:
    optim: Optional[str] = field(default='adamw_torch')
    group_by_data_type: Optional[bool] = field(default=True)
    merge_adapter: Optional[bool] = field(default=False)
    lora_enable: Optional[bool] = field(default=False)
    lora_type: Optional[str] = field(default='qkvo')
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_bias: Optional[str] = field(default='none')
    lora_lr: Optional[float] = field(default=None)
    head_lr: Optional[float] = field(default=None)
    tuning_modules: Optional[str] = field(default=None)
    save_full_model: Optional[bool] = field(default=False)
    remove_unused_columns: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(CustomArguments, HFTrainingArguments):
    pass


def get_target_modules(model, lora_type, base_model):
    lora_modes = lora_type.split('_')
    assert all(mode in ('qkvo', 'linear', 'all') for mode in lora_modes)

    if base_model not in SUPPORTED_BASE_MODELS:
        raise ValueError(f'unknown base model: {base_model}')

    qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']
    target_modules = set()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if 'all' not in lora_modes and 'visual' in name:
            continue
        if 'qkvo' in lora_modes and not any(name.endswith(key) for key in qkvo_keys):
            continue
        target_modules.add(name)

    return target_modules


def _resolve_dtype(training_args):
    return torch.bfloat16 if training_args.bf16 else torch.float32


def _build_model_and_processor(model_args, training_args):
    if model_args.base_model not in MODELS:
        raise ValueError(
            f"unsupported base model: {model_args.base_model}. "
            "Qwen2.5-VL requires transformers>=4.50.0 and an installed qwen2_5_vl backend.")

    config_cls, model_cls = MODELS[model_args.base_model]
    dtype = _resolve_dtype(training_args)

    config = config_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype)
    config.update(vars(model_args))

    if config.model_type in AGENT_MODEL_TYPES:
        model, processor = build_model(
            model_args.model_name_or_path,
            config=config,
            is_trainable=True,
            merge_adapter=training_args.merge_adapter,
            dtype=dtype)
        return model, processor

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, do_resize=False)
    model = model_cls.from_pretrained(model_args.model_name_or_path, config=config, attn_implementation='sdpa')

    model.config.base_model_path = model_args.model_name_or_path
    model.reset_conv_parameters()
    model.requires_grad_(False)
    return model, processor


def _maybe_enable_lora(model, model_args, training_args):
    if not training_args.lora_enable or isinstance(model, PeftModel):
        return model

    target_modules = get_target_modules(model, training_args.lora_type, model.config.base_model)
    tune_lm_head = model.config.role in ('all_in_one', 'grounder', 'checker')
    print(f'LoRA target modules: {target_modules}')

    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        target_modules=target_modules,
        modules_to_save=['embed_tokens', 'lm_head'] if tune_lm_head else None)
    return get_peft_model(model, lora_config, adapter_name=model_args.role)


def _register_special_tokens(model, processor):
    new_tokens = processor.tokenizer.add_special_tokens(
        dict(additional_special_tokens=[REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN]))
    print(f'Added {new_tokens} new token(s)')

    model.config.reg_token_id = processor.tokenizer.convert_tokens_to_ids(REG_TOKEN)
    model.config.seg_s_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_S_TOKEN)
    model.config.seg_e_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_E_TOKEN)

    if new_tokens <= 0 or len(processor.tokenizer) <= model.config.vocab_size:
        return new_tokens

    print(f'Expanding vocab size: {model.config.vocab_size} -> {len(processor.tokenizer)}')
    model.resize_token_embeddings(len(processor.tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_embeddings[-new_tokens:] = input_embeddings[:-new_tokens].mean(0, keepdim=True)
    output_embeddings[-new_tokens:] = output_embeddings[:-new_tokens].mean(0, keepdim=True)
    return new_tokens


def _mark_trainable_parameters(model, model_args, training_args, new_tokens):
    tuning_modules = [] if training_args.tuning_modules is None else training_args.tuning_modules.split(',')

    for name, param in model.named_parameters():
        if not training_args.lora_enable and new_tokens > 0 and any(key in name for key in ('embed_tokens', 'lm_head')):
            param.requires_grad = True

        if 'projector' in tuning_modules and 'visual.merger' in name:
            param.requires_grad = True

        if model_args.role in ('all_in_one', 'grounder') and any(key in name for key in HEAD_KEYS):
            param.requires_grad = True


def _log_parameter_summary(model, processor):
    for name, param in model.named_parameters():
        print(param.requires_grad, param.dtype, param.shape, name)

    total_params = sum(param.numel() for param in model.parameters())
    learnable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
    print(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

    input_size = model.get_input_embeddings().num_embeddings
    output_size = model.get_output_embeddings().out_features
    assert input_size == output_size, (input_size, output_size)
    print(f'Tokenizer size: {len(processor.tokenizer)} Vocab size: {model.config.vocab_size} Embed size: {input_size}')


def main(training_args_cls=TrainingArguments, trainer_cls=CustomTrainer):
    parser = HfArgumentParser((ModelArguments, DataArguments, training_args_cls))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert model_args.role in ('all_in_one', 'planner', 'grounder', 'checker', 'answerer')

    model, processor = _build_model_and_processor(model_args, training_args)
    model = _maybe_enable_lora(model, model_args, training_args)
    new_tokens = _register_special_tokens(model, processor)
    _mark_trainable_parameters(model, model_args, training_args, new_tokens)

    if training_args.local_rank in (0, -1):
        _log_parameter_summary(model, processor)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=HybridDataCollator(processor.tokenizer),
        train_dataset=HybridDataset(processor, model.config, model_args, data_args, training_args),
        processor=processor,
        head_keys=HEAD_KEYS)

    has_ckpt = bool(nncore.find(training_args.output_dir, 'checkpoint-*'))
    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_state()
    trainer.gather_and_save_model()


if __name__ == '__main__':
    main()
