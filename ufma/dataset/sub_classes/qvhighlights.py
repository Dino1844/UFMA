# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import nncore

from ufma.dataset.hybrid import DATASETS
from ufma.dataset.wrappers import GroundingDataset
from ufma.utils.parser import parse_query


@DATASETS.register(name='qvhighlights')
class QVHighlightsDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/qvhighlights/highlight_train_release.jsonl'
    ANNO_PATH_VALID = 'data/qvhighlights/highlight_val_release.jsonl'
    ANNO_PATH_TEST = 'data/qvhighlights/highlight_test_release.jsonl'

    VIDEO_ROOT = 'data/qvhighlights/videos_3fps_480_noaudio'

    UNIT = 2.0

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            print('WARNING: Test split does not have ground truth annotations')
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['vid']
            qid = raw_anno['qid']

            anno = dict(
                source='qvhighlights',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno.get('relevant_windows'),
                vid=vid,
                qid=qid)

            annos.append(anno)

        return annos


@DATASETS.register(name='qvhighlights_single')
class QVHighlightsSingleDataset(QVHighlightsDataset):

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH_TRAIN)

        annos = []
        for raw_anno in raw_annos:
            # skip samples with multiple moments
            if len(raw_anno['relevant_windows']) > 1:
                continue

            vid = raw_anno['vid']

            anno = dict(
                source='qvhighlights_single',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno.get('relevant_windows'))

            annos.append(anno)

        return annos
