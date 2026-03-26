# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import argparse
import copy

import nncore
import torch

from ufma.constants import GROUNDER_PROMPT
from ufma.dataset.hybrid import DATASETS
from ufma.dataset.utils import process_vision_info
from ufma.model.builder import build_model
from ufma.utils.io import get_duration

DEFAULT_GENERATE_KWARGS = dict(
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=None,
    max_new_tokens=256)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_gnd_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def _resolve_output_path(base_path, chunk, index):
    filename = f'output_{index}.jsonl' if chunk > 1 else 'output.jsonl'
    return nncore.join(base_path, filename)


def _slice_chunk(items, chunk, index):
    return [items[i::chunk] for i in range(chunk)][index]


def _build_messages(video_path, query, num_threads):
    return [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'num_threads': num_threads,
            'min_pixels': 36 * 28 * 28,
            'max_pixels': 64 * 28 * 28,
            'max_frames': 150,
            'fps': 1.0
        }, {
            'type': 'text',
            'text': GROUNDER_PROMPT.format(query)
        }]
    }]


def _prepare_batch(processor, messages, device):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(text)
    images, videos = process_vision_info(messages)
    return processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)


def _decode_response(processor, data, output_ids):
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids.numel() > 0 and output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    return processor.decode(output_ids, clean_up_tokenization_spaces=False)


def _normalize_predictions(blob, duration, dataset_cls):
    pred = blob[:, :2] * duration
    conf = blob[:, 2:]

    pred = pred.clamp(min=0, max=duration)
    unit = getattr(dataset_cls, 'UNIT', 0.001)
    pred = torch.round(pred / unit).long() * unit

    inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
    pred[inds] = pred[inds].roll(1)
    return torch.cat((pred, conf), dim=1)


def main():
    args = parse_args()
    pred_path = _resolve_output_path(args.pred_path, args.chunk, args.index)

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    dataset_cls = DATASETS.get(args.dataset)
    annos = _slice_chunk(dataset_cls.load_annos(split=args.split), args.chunk, args.index)

    dumps = []
    for idx in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[idx])
        dump = {}
        video_path = anno['video_path']
        query = anno['query']
        duration = anno.get('duration')
        span = anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)

        print()
        print(video_path)
        print(duration)
        print(query)

        messages = _build_messages(video_path, query, args.num_threads)
        data = _prepare_batch(processor, messages, device)
        output_ids = model.generate(**data, **DEFAULT_GENERATE_KWARGS)
        response = _decode_response(processor, data, output_ids)
        print(response)

        if len(model.reg) > 0:
            pred = _normalize_predictions(model.reg[0].cpu().float(), duration, dataset_cls)
        else:
            print('WARNING: Failed to parse grounder response')
            pred = torch.Tensor([[0, duration, 1]])

        print(pred[0], span, duration)

        dump['vid'] = anno['vid']
        dump['qid'] = anno['qid']
        dump['pred_relevant_windows'] = pred.tolist()
        dumps.append(dump)

    nncore.dump(dumps, pred_path)


if __name__ == '__main__':
    main()
