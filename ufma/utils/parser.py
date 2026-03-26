# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import re

WHITESPACE_RE = re.compile(r'\s+')


def _clamp(value, lower, upper):
    return min(upper, max(lower, value))


def _normalize_text(text, strip_trailing_period=False):
    normalized = WHITESPACE_RE.sub(' ', text).strip()
    if strip_trailing_period:
        normalized = normalized.rstrip('.').strip()
    return normalized


def parse_span(span, duration, min_len=-1):
    start, end = span
    start = _clamp(start, 0, duration)
    end = _clamp(end, 0, duration)
    start, end = min(start, end), max(start, end)

    if min_len != -1 and end - start < min_len:
        half_length = min_len / 2
        center = _clamp((start + end) / 2, half_length, max(half_length, duration - half_length))
        start, end = center - half_length, center + half_length

    return _clamp(start, 0, duration), _clamp(end, 0, duration)


def parse_query(query):
    return _normalize_text(query, strip_trailing_period=True)


def parse_question(question):
    return _normalize_text(question)
