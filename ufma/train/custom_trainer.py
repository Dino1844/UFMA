# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import warnings

import nncore
import torch
from deepspeed import zero
from safetensors.torch import load_model, save_file
from torch.utils.data import Sampler
from transformers import Trainer, TrainerCallback
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import CHAT_TEMPLATE_NAME


def gather_param(param):
    if hasattr(param, 'ds_id'):
        with zero.GatheredParameters([param]):
            return param.data.detach().cpu().clone()
    return param.detach().cpu().clone()


def gather_lora_params(model, bias):
    assert bias in ('lora_only', 'all', 'none')

    if bias == 'lora_only':
        state_dict = {}
        maybe_bias = {}
        lora_bias_names = set()

        for name, param in model.named_parameters():
            if 'modules_to_save' in name or 'lora_' in name:
                state_dict[name] = param
            elif 'bias' in name:
                maybe_bias[name] = param

            if 'lora_' in name:
                lora_bias_names.add(name.split('lora_')[0] + 'bias')

        for name, param in maybe_bias.items():
            if name in lora_bias_names:
                state_dict[name] = param
    else:
        keys = ['lora_', 'modules_to_save', 'bias'] if bias == 'all' else ['lora_', 'modules_to_save']
        state_dict = {name: param for name, param in model.named_parameters() if any(key in name for key in keys)}

    return {name: gather_param(param) for name, param in state_dict.items()}


def gather_key_params(model, keys):
    state_dict = {
        name: param
        for name, param in model.named_parameters() if param.requires_grad and any(key in name for key in keys)
    }
    return {name: gather_param(param) for name, param in state_dict.items()}


class GroupSampler(Sampler):

    def __init__(self, group_size, data_types, seed):
        self.group_size = group_size
        self.data_types = data_types
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.data_types)

    def _build_groups(self):
        unique_types = []
        grouped_indices = []

        for index, data_type in enumerate(self.data_types):
            if data_type not in unique_types:
                unique_types.append(data_type)
                grouped_indices.append([])
            grouped_indices[unique_types.index(data_type)].append(index)

        return grouped_indices

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        grouped_batches = []
        for group in self._build_groups():
            shuffled = [group[idx] for idx in torch.randperm(len(group), generator=generator)]
            batches = [shuffled[i:i + self.group_size] for i in range(0, len(shuffled), self.group_size)]
            if batches and len(batches[-1]) < self.group_size:
                batches = batches[:-1]
            grouped_batches.extend(batches)

        batch_order = torch.randperm(len(grouped_batches), generator=generator)
        indices = [idx for batch_idx in batch_order for idx in grouped_batches[batch_idx]]
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


class SetEpochCallback(TrainerCallback):

    # partially fixed in https://github.com/huggingface/accelerate/pull/3246
    # but not for the case of batch_sampler.batch_sampler.sampler
    def on_epoch_begin(self, args, state, control, **kwargs):
        shard_sampler = kwargs['train_dataloader'].batch_sampler
        batch_sampler = getattr(shard_sampler, 'batch_sampler', shard_sampler)
        batch_sampler.sampler.set_epoch(int(state.epoch))


class CustomTrainer(Trainer):

    def __init__(self, *args, processor=None, head_keys=None, **kwargs):
        super().__init__(*args, tokenizer=processor, **kwargs)
        self.add_callback(SetEpochCallback())
        self.processor = processor
        self.head_keys = head_keys or []

    def _get_train_sampler(self):
        if not self.args.group_by_data_type:
            return super()._get_train_sampler()

        return GroupSampler(
            self.args.train_batch_size * self.args.world_size,
            self.train_dataset.data_types,
            self.args.seed)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        model = self.model if model is None else model
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        partial_path = nncore.join(resume_from_checkpoint, 'pytorch_model.safetensors')
        if nncore.is_file(partial_path):
            load_model(model, partial_path, strict=False, device=model.device)

    def _get_decay_parameter_names(self):
        decay_names = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        return {name for name in decay_names if 'bias' not in name}

    def _split_trainable_parameters(self):
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def _resolve_group_lrs(self):
        lora_lr = self.args.learning_rate if self.args.lora_lr is None else self.args.lora_lr
        head_lr = self.args.learning_rate if self.args.head_lr is None else self.args.head_lr
        return lora_lr, head_lr

    def _build_optimizer_groups(self):
        grad_params = self._split_trainable_parameters()
        decay_names = self._get_decay_parameter_names()
        lora_lr, head_lr = self._resolve_group_lrs()

        lora_names = {name for name, _ in grad_params if 'lora' in name}
        head_names = {name for name, _ in grad_params if any(key in name for key in self.head_keys)}
        assert not (lora_names & head_names), 'LoRA params and head params should be disjoint'

        group_specs = [
            dict(name='base_decay', use_decay=True, use_lora=False, use_head=False, lr=None),
            dict(name='base_nodecay', use_decay=False, use_lora=False, use_head=False, lr=None),
            dict(name='lora_decay', use_decay=True, use_lora=True, use_head=False, lr=lora_lr),
            dict(name='lora_nodecay', use_decay=False, use_lora=True, use_head=False, lr=lora_lr),
            dict(name='head_decay', use_decay=True, use_lora=False, use_head=True, lr=head_lr),
            dict(name='head_nodecay', use_decay=False, use_lora=False, use_head=True, lr=head_lr),
        ]

        optimizer_groups = []
        for spec in group_specs:
            params = []
            for name, param in grad_params:
                in_decay = name in decay_names
                in_lora = name in lora_names
                in_head = name in head_names

                if in_decay != spec['use_decay']:
                    continue
                if in_lora != spec['use_lora']:
                    continue
                if in_head != spec['use_head']:
                    continue
                params.append(param)

            if not params:
                continue

            group = {
                'params': params,
                'weight_decay': self.args.weight_decay if spec['use_decay'] else 0.0,
            }
            if spec['lr'] is not None:
                group['lr'] = spec['lr']
            optimizer_groups.append(group)

        return optimizer_groups

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_groups = self._build_optimizer_groups()
            optim_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optim_cls(optimizer_groups, **kwargs)
        return self.optimizer

    def _is_zero3_enabled(self):
        config = getattr(self.accelerator, 'deepspeed_config', None)
        if config is None:
            return False
        return config['zero_optimization']['stage'] == 3

    def _save_processor_bundle(self, output_dir):
        if self.processor is None or not self.args.should_save:
            return

        self.processor.save_pretrained(output_dir)
        if self.processor.chat_template is not None:
            nncore.dump({'chat_template': self.processor.chat_template}, nncore.join(output_dir, CHAT_TEMPLATE_NAME),
                        indent=2)

    def _save_config_bundle(self, output_dir):
        if not self.args.should_save:
            return

        self.model.config.save_pretrained(output_dir)
        self.model.generation_config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _save_partial_weights(self, output_dir):
        state_dict = gather_key_params(self.model, self.head_keys)
        if self.args.should_save and state_dict:
            save_file(state_dict, nncore.join(output_dir, 'pytorch_model.safetensors'))

    def gather_and_save_model(self):
        deepspeed_zero3 = self._is_zero3_enabled()
        output_dir = self.args.output_dir

        if self.args.should_save:
            print(f'Saving final model to {nncore.abs_path(output_dir)}...')

        self._save_processor_bundle(output_dir)

        if self.args.save_full_model and self.args.lora_enable and deepspeed_zero3:
            warnings.warn('LoRA models cannot be saved in full mode under zero3, saving adapters instead')
            self.args.save_full_model = False

        if self.args.save_full_model:
            if self.args.lora_enable:
                self.model = self.model.merge_and_unload()

            if deepspeed_zero3 and not self.model_wrapped.zero_gather_16bit_weights_on_model_save():
                warnings.warn('Saving zero checkpoint, use zero_to_fp32.py to recover weights')
                self.model_wrapped.save_checkpoint(output_dir)
                return

            if deepspeed_zero3:
                state_dict = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            else:
                state_dict = self.model.state_dict()

            if self.args.should_save:
                state_dict = {key[17:] if key.startswith('base_model.model.') else key: value for key, value in state_dict.items()}
                self._save(output_dir, state_dict=state_dict)
            return

        if self.args.lora_enable:
            state_dict = gather_lora_params(self.model, self.args.lora_bias)
            if self.args.should_save:
                self.model.save_pretrained(output_dir, state_dict=state_dict)

        self._save_config_bundle(output_dir)
        self._save_partial_weights(output_dir)

    def _save_checkpoint(self, model, trial, **kwargs):
        output_dir = self._get_output_dir(trial)
        output_dir = nncore.join(output_dir, f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')

        if self.args.should_save:
            print(f'Saving checkpoint to {nncore.abs_path(output_dir)}...')

        super()._save_checkpoint(model, trial, **kwargs)
        self._save_processor_bundle(output_dir)

        if self.args.lora_enable:
            if self.args.should_save:
                self.model.config.save_pretrained(output_dir)
            self._save_partial_weights(output_dir)
