# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import pysrt
from decord import VideoReader


def time_to_seconds(timestamp):
    return (timestamp.hour * 60 + timestamp.minute) * 60 + timestamp.second + timestamp.microsecond / 1_000_000


def load_subtitle(path):
    subtitles = []
    for subtitle in pysrt.open(path):
        subtitles.append((
            time_to_seconds(subtitle.start.to_time()),
            time_to_seconds(subtitle.end.to_time()),
            subtitle.text,
        ))
    return subtitles


def get_duration(path, num_threads=1):
    if isinstance(path, list):
        return len(path)

    video_reader = VideoReader(path, num_threads=num_threads)
    return len(video_reader) / video_reader.get_avg_fps()
