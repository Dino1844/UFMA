# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch

from ufma.constants import GROUNDER_PROMPT, PLANNER_PROMPT, CHECKER_PROMPT
from ufma.dataset.hybrid import DATASETS
from ufma.dataset.utils import process_vision_info
from ufma.model.builder import build_model
from ufma.utils.io import get_duration, load_subtitle
from ufma.utils.parser import parse_query, parse_span

YES_TOKEN_ID = 9454
NO_TOKEN_ID = 2753
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
    parser.add_argument('--model_checker_path')
    parser.add_argument('--model_pla_path')
    parser.add_argument('--model_ans_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checker_topk', type=int, default=5)
    parser.add_argument('--obj_lambda', type=float, default=1.0)
    parser.add_argument('--obj_beta', type=float, default=0.15)
    parser.add_argument('--cd_rounds', type=int, default=2)
    parser.add_argument('--cd_step_ratio', type=float, default=0.2)
    parser.add_argument('--cd_min_step', type=float, default=0.5)
    parser.add_argument('--uncert_margin_thr', type=float, default=0.03)
    parser.add_argument('--uncert_disagree_thr', type=float, default=0.08)
    parser.add_argument('--active_dense_ratio', type=float, default=0.5)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


def _resolve_output_path(base_path, chunk, index, filename='output.json'):
    return nncore.join(base_path, f'output_{index}.json' if chunk > 1 else filename)


def _slice_chunk(items, chunk, index):
    return [items[i::chunk] for i in range(chunk)][index]


def _activate_adapter(model, adapter_name):
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter(adapter_name)


def _load_role_adapter(model, model_path, role_name):
    if model_path is None:
        return False

    adapter_path = nncore.join(model_path, role_name)
    if not nncore.is_dir(adapter_path):
        return False

    print(f'Initializing role *{role_name}*')
    model.load_adapter(adapter_path, adapter_name=role_name)
    return True


def _build_video_message(video_path, prompt, num_threads, **video_kwargs):
    video_content = {
        'type': 'video',
        'video': video_path,
        'num_threads': num_threads,
        **video_kwargs,
    }
    return [{
        'role': 'user',
        'content': [
            video_content,
            {
                'type': 'text',
                'text': prompt,
            },
        ],
    }]


def _prepare_batch(processor, messages, device, prompt_suffix=''):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    if prompt_suffix:
        text += prompt_suffix
    print(text)

    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    return text, data.to(device)


def _decode_generated_text(processor, data, output_ids):
    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids.numel() > 0 and output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    return processor.decode(output_ids, clean_up_tokenization_spaces=False)


def _generate_response(model, processor, messages, device, adapter_name=None, prompt_suffix='', disable_adapter=False):
    _, data = _prepare_batch(processor, messages, device, prompt_suffix=prompt_suffix)

    if adapter_name is not None:
        _activate_adapter(model, adapter_name)
        context = nullcontext()
    elif disable_adapter:
        context = model.disable_adapter()
    else:
        context = nullcontext()

    with context:
        output_ids = model.generate(**data, **DEFAULT_GENERATE_KWARGS)

    response = _decode_generated_text(processor, data, output_ids)
    print(response)
    return response


def _format_multiple_choice_prompt(question, options, style):
    if style == 'direct':
        return question

    prompt = question + '\nOptions:'
    for idx, opt in enumerate(options):
        prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
    return prompt + '\nPlease only give the best option.'


def _normalize_grounding_predictions(blob, duration, dataset_cls):
    pred = blob[:, :2] * duration
    conf = blob[:, -1].tolist()

    pred = pred.clamp(min=0, max=duration)
    unit = getattr(dataset_cls, 'UNIT', 0.001)
    pred = torch.round(pred / unit).long() * unit

    inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
    pred[inds] = pred[inds].roll(1)
    return pred.tolist(), conf


def _fallback_grounding_predictions(duration, use_checker):
    if use_checker:
        return [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)], [0] * 5
    return [[0, duration]], [0]


def _select_answer_span(selected, duration, dataset_cls):
    if hasattr(dataset_cls, 'MIN_RATIO'):
        min_len = duration * dataset_cls.MIN_RATIO
    else:
        min_len = getattr(dataset_cls, 'MIN_LEN', 32)
    return parse_span(selected, duration, min_len)


def _build_checker(model, processor, device, video_path, question, duration, num_threads=1):
    cache = dict()

    def _score(cand):
        s0, e0 = parse_span(cand, duration, 2)
        key = (round(s0, 3), round(e0, 3))
        if key in cache:
            return cache[key]

        offset = (e0 - s0) / 2
        s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

        deno = max(e1 - s1, 1e-6)
        s = (s0 - s1) / deno
        e = (e0 - s1) / deno

        messages = _build_video_message(
            video_path,
            CHECKER_PROMPT.format(question),
            num_threads,
            video_start=s1,
            video_end=e1,
            min_pixels=36 * 28 * 28,
            max_pixels=64 * 28 * 28,
            max_frames=64,
            fps=2.0)

        _, data = _prepare_batch(processor, messages, device)

        video_grid_thw = data['video_grid_thw'][0]
        num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
        assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

        pos_s, pos_e = round(s * num_frames), round(e * num_frames)
        pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
        assert pos_s <= pos_e, (num_frames, s, e)

        base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
        pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

        input_ids = data['input_ids'][0].tolist()
        input_ids.insert(pos_s, model.config.seg_s_token_id)
        input_ids.insert(pos_e, model.config.seg_e_token_id)
        data['input_ids'] = torch.LongTensor([input_ids])
        data['attention_mask'] = torch.ones_like(data['input_ids'])
        data = data.to(device)

        with torch.inference_mode():
            logits = model(**data).logits[0, -1].softmax(dim=-1)

        support = logits[YES_TOKEN_ID].item()
        counter = logits[NO_TOKEN_ID].item()
        score = (logits[YES_TOKEN_ID] - logits[NO_TOKEN_ID]).sigmoid().item()

        out = dict(span=[s0, e0], support=support, counter=counter, score=score)
        cache[key] = out
        return out

    return _score, cache


def _compute_objective(span, anchor_span, duration, support, counter, score_lambda=1.0, jitter_beta=0.15):
    jitter = 0.0
    if anchor_span is not None:
        s, e = span
        a_s, a_e = parse_span(anchor_span, duration, 2)
        anchor_len = max(a_e - a_s, duration * 0.05, 1e-6)
        jitter = (abs(s - a_s) + abs(e - a_e)) / (2 * anchor_len)
    objective = -support + score_lambda * counter + jitter_beta * jitter
    return objective, jitter


def _score_with_objective(cand, checker, duration, anchor_span=None, score_lambda=1.0, jitter_beta=0.15):
    s, e = parse_span(cand, duration, 2)
    scored = checker([s, e])
    objective, jitter = _compute_objective([s, e], anchor_span, duration, scored['support'], scored['counter'],
                                           score_lambda=score_lambda, jitter_beta=jitter_beta)
    return dict(
        span=[s, e],
        support=scored['support'],
        counter=scored['counter'],
        score=scored['score'],
        jitter=jitter,
        objective=objective)


def _estimate_uncertainty(scored, duration, margin_thr=0.03, disagree_thr=0.08):
    if len(scored) == 0:
        return dict(
            active=False,
            focus='both',
            by_margin=False,
            by_disagree=False,
            margin=float('inf'),
            start_spread=0.0,
            end_spread=0.0,
            disagree_ratio=0.0)

    if len(scored) == 1:
        return dict(
            active=False,
            focus='both',
            by_margin=False,
            by_disagree=False,
            margin=float('inf'),
            start_spread=0.0,
            end_spread=0.0,
            disagree_ratio=0.0)

    margin = scored[1]['objective'] - scored[0]['objective']
    top_k = min(3, len(scored))
    starts = [m['span'][0] for m in scored[:top_k]]
    ends = [m['span'][1] for m in scored[:top_k]]
    start_spread = max(starts) - min(starts)
    end_spread = max(ends) - min(ends)
    disagree_ratio = max(start_spread, end_spread) / max(duration, 1e-6)

    by_margin = margin <= margin_thr
    by_disagree = disagree_ratio >= disagree_thr

    if by_margin and not by_disagree:
        focus = 'both'
    else:
        focus = 'start' if start_spread >= end_spread else 'end'

    return dict(
        active=by_margin or by_disagree,
        focus=focus,
        by_margin=by_margin,
        by_disagree=by_disagree,
        margin=margin,
        start_spread=start_spread,
        end_spread=end_spread,
        disagree_ratio=disagree_ratio)


def _coordinate_descent_optimize(init_span,
                                 duration,
                                 scorer,
                                 rounds=2,
                                 step_ratio=0.2,
                                 min_step=0.5,
                                 uncertainty=None,
                                 active_dense_ratio=0.5):
    current = scorer(init_span)
    history = [dict(round=0, stage='init', **current)]

    s0, e0 = current['span']
    base_step = max(min_step, (e0 - s0) * step_ratio)

    uncertainty = uncertainty or dict(active=False, focus='both', start_spread=0.0, end_spread=0.0)

    for rid in range(rounds):
        step = max(min_step / 4, base_step / (2**rid))

        offsets = [-step, 0.0, step]
        if uncertainty['active'] and uncertainty['focus'] in ('end', 'both'):
            spread = uncertainty.get('end_spread', 0.0)
            dense = max(step * active_dense_ratio, spread * 0.5, min_step / 2)
            offsets += [-dense / 2, dense / 2]
        offsets = sorted(set(round(off, 6) for off in offsets))

        cands = [[current['span'][0], current['span'][1] + off] for off in offsets]
        current = min([scorer(c) for c in cands], key=lambda x: x['objective'])
        history.append(dict(round=rid + 1, stage='opt_end', **current))

        offsets = [-step, 0.0, step]
        if uncertainty['active'] and uncertainty['focus'] in ('start', 'both'):
            spread = uncertainty.get('start_spread', 0.0)
            dense = max(step * active_dense_ratio, spread * 0.5, min_step / 2)
            offsets += [-dense / 2, dense / 2]
        offsets = sorted(set(round(off, 6) for off in offsets))

        cands = [[current['span'][0] + off, current['span'][1]] for off in offsets]
        current = min([scorer(c) for c in cands], key=lambda x: x['objective'])
        history.append(dict(round=rid + 1, stage='opt_start', **current))

    return current, history


def main():
    args = parse_args()

    pred_path = _resolve_output_path(args.pred_path, args.chunk, args.index)

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    # NOTE:
    # 1. grounder is always true so no need to store
    # 2. answerer would always be used (when set to false, the base model would be used as the answerer)
    adapter_state = dict(planner=False, checker=False, answerer=False)

    print('Initializing role *grounder*')
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    adapter_state['planner'] = _load_role_adapter(model, args.model_pla_path, 'planner')
    adapter_state['checker'] = _load_role_adapter(model, args.model_checker_path, 'checker')
    adapter_state['answerer'] = _load_role_adapter(model, args.model_ans_path, 'answerer')

    dataset_cls = DATASETS.get(args.dataset)

    annos = dataset_cls.load_annos(split=args.split)
    annos = _slice_chunk(annos, args.chunk, args.index)

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        print()
        print(video_path)
        print(duration)

        # sometimes the sample is for grounding only
        do_answering = all(k in anno for k in ('question', 'options'))

        if do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']
            prompt = _format_multiple_choice_prompt(question, options, args.style)

            print(prompt)
            print(options)
            print(ans)
        else:
            question = anno['query']
            print(question)

        # do grounding by default
        do_grounding = True

        # initialize grounding query as question
        query = question

        # initialize agent list
        dump['agents'] = []

        if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
            print('=============== planner ===============')

            dump['agents'].append('planner')

            messages = _build_video_message(
                video_path,
                PLANNER_PROMPT.format(question),
                args.num_threads,
                min_pixels=36 * 28 * 28,
                max_pixels=64 * 28 * 28,
                max_frames=100,
                fps=1.0)

            response = _generate_response(model, processor, messages, device, adapter_name='planner')

            dump['planner_response'] = response

            try:
                parsed = json.loads(response)
                action = parsed[0] if isinstance(parsed, list) else parsed
                if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                    query = action['value']
                    dump['planner_parsed_query'] = query
                elif args.auto_planning and action['type'].lower() == 'answerer':
                    do_grounding = False
            except Exception:
                print('WARNING: Failed to parse planner response')

        if do_grounding:
            print('=============== grounder ===============')

            dump['agents'].append('grounder')

            query = parse_query(query)

            messages = _build_video_message(
                video_path,
                GROUNDER_PROMPT.format(query),
                args.num_threads,
                min_pixels=36 * 28 * 28,
                max_pixels=64 * 28 * 28,
                max_frames=150,
                fps=1.0)

            response = _generate_response(model, processor, messages, device, adapter_name='grounder')

            dump['grounder_response'] = response
            dump['grounder_success'] = len(model.reg) > 0

            if dump['grounder_success']:
                pred, conf = _normalize_grounding_predictions(model.reg[0].cpu().float(), duration, dataset_cls)
            else:
                print('WARNING: Failed to parse grounder response')
                pred, conf = _fallback_grounding_predictions(duration, adapter_state['checker'])

            print(pred[0], span, duration)
            dump['pred'] = pred
            dump['conf'] = conf

        if do_grounding and adapter_state['checker'] and len(pred) > 1:
            print('=============== checker ===============')

            dump['agents'].append('checker')
            _activate_adapter(model, 'checker')

            checker, checker_cache = _build_checker(
                model=model,
                processor=processor,
                device=device,
                video_path=video_path,
                question=question,
                duration=duration,
                num_threads=args.num_threads)

            topk = max(1, min(args.checker_topk, len(pred)))
            coarse = []
            probs = []

            for idx, cand in enumerate(pred[:topk]):
                stat = _score_with_objective(
                    cand,
                    checker=checker,
                    duration=duration,
                    score_lambda=args.obj_lambda,
                    jitter_beta=0)
                stat['idx'] = idx
                coarse.append(stat)
                probs.append(stat['score'])

            coarse = sorted(coarse, key=lambda x: x['objective'])
            ranks = [m['idx'] for m in coarse]
            print(probs)
            print(ranks)

            pred = [pred[idx] for idx in ranks] + pred[topk:]
            conf = [conf[idx] for idx in ranks] + conf[topk:]

            uncertainty = _estimate_uncertainty(
                coarse,
                duration=duration,
                margin_thr=args.uncert_margin_thr,
                disagree_thr=args.uncert_disagree_thr)

            anchor_span = coarse[0]['span']
            scorer = lambda x: _score_with_objective(
                x,
                checker=checker,
                duration=duration,
                anchor_span=anchor_span,
                score_lambda=args.obj_lambda,
                jitter_beta=args.obj_beta)

            refined, traj = _coordinate_descent_optimize(
                anchor_span,
                duration=duration,
                scorer=scorer,
                rounds=args.cd_rounds,
                step_ratio=args.cd_step_ratio,
                min_step=args.cd_min_step,
                uncertainty=uncertainty,
                active_dense_ratio=args.active_dense_ratio)

            pred[0] = refined['span']
            conf[0] = max(conf[0], refined['support'])
            print(pred[0], span, duration)

            dump['probs'] = probs
            dump['ranks'] = ranks
            dump['pred_ori'] = dump['pred']
            dump['conf_ori'] = dump['conf']
            dump['pred'] = pred
            dump['conf'] = conf
            dump['checker_support'] = [m['support'] for m in coarse]
            dump['checker_counter'] = [m['counter'] for m in coarse]
            dump['checker_objective'] = [m['objective'] for m in coarse]
            dump['checker_uncertainty'] = uncertainty
            dump['checker_fine'] = refined
            dump['checker_fine_traj'] = traj
            dump['checker_calls'] = len(checker_cache)

        if do_answering:
            print('=============== answerer ===============')

            dump['agents'].append('answerer')

            # choose the potential best moment
            selected = pred[0] if 'pred' in dump else [0, duration]
            s, e = _select_answer_span(selected, duration, dataset_cls)
            print([s, e], span, duration)

            if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
                subs = load_subtitle(anno['subtitle_path'])
                subs = [f'{round(a - s, 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a >= s and b <= e]
                # use only the first 100 subtitles to save memory
                subs = ''.join(subs[:100])
                prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

            messages = _build_video_message(
                video_path,
                prompt,
                args.num_threads,
                video_start=s,
                video_end=e,
                min_pixels=getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                max_pixels=getattr(dataset_cls, 'MAX_PIXELS', 256) * 28 * 28,
                max_frames=getattr(dataset_cls, 'MAX_FRAMES', 32),
                fps=getattr(dataset_cls, 'FPS', 2.0))

            response = _generate_response(
                model,
                processor,
                messages,
                device,
                adapter_name='answerer' if adapter_state['answerer'] else None,
                prompt_suffix='Best Option: (' if args.style == 'mcq' else '',
                disable_adapter=not adapter_state['answerer'])

            dump['answerer_response'] = response
            dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)


if __name__ == '__main__':
    main()
