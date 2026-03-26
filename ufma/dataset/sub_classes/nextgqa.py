# Copyright (c) 2025, VideoMind Team. Licensed under the BSD-3-Clause License.
# Modifications Copyright (c) 2026, UFMA Authors.

import csv

import nncore

from ufma.dataset.hybrid import DATASETS
from ufma.dataset.wrappers import AnsweringCropDataset, AnsweringDataset, GroundingDataset
from ufma.utils.parser import parse_query, parse_question


@DATASETS.register(name='nextgqa')
class NExTGQADataset(AnsweringDataset):

    ANNO_PATH_VALID = 'data/nextgqa/val.csv'
    ANNO_PATH_TEST = 'data/nextgqa/test.csv'

    SPAN_PATH_VALID = 'data/nextgqa/gsub_val.json'
    SPAN_PATH_TEST = 'data/nextgqa/gsub_test.json'

    VIDEO_ID_MAP = 'data/nextgqa/map_vid_vidorID.json'
    VIDEO_ROOT = 'data/nextqa/videos'

    SOURCE = 'nextgqa'
    DATA_TYPE = 'multimodal'

    UNIT = 0.1

    @classmethod
    def load_annos(self, split='valid'):
        assert split in ('valid', 'test')

        if split == 'valid':
            anno_path = self.ANNO_PATH_VALID
            raw_spans = nncore.load(self.SPAN_PATH_VALID)
        else:
            anno_path = self.ANNO_PATH_TEST
            raw_spans = nncore.load(self.SPAN_PATH_TEST)

        with open(anno_path, mode='r') as f:
            reader = csv.DictReader(f)
            raw_annos = [d for d in reader]

        video_id_map = nncore.load(self.VIDEO_ID_MAP)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']
            qid = raw_anno['qid']

            video_id = video_id_map[vid]

            question = raw_anno['question'][0].upper() + raw_anno['question'][1:] + '?'

            query = parse_query(question)
            question = parse_question(question)
            options = [raw_anno[k][0].upper() + raw_anno[k][1:] for k in ('a0', 'a1', 'a2', 'a3', 'a4')]
            answer = raw_anno['answer'][0].upper() + raw_anno['answer'][1:]
            ans = chr(ord('A') + options.index(answer))

            anno = dict(
                source=self.SOURCE,
                data_type=self.DATA_TYPE,
                video_path=nncore.join(self.VIDEO_ROOT, video_id + '.mp4'),
                duration=raw_spans[vid]['duration'],
                query=query,
                question=question,
                options=options,
                answer=answer,
                ans=ans,
                span=raw_spans[vid]['location'][qid],
                task=raw_anno['type'])

            annos.append(anno)

        return annos


@DATASETS.register(name='nextgqa_crop')
class NExTGQACropDataset(AnsweringCropDataset, NExTGQADataset):

    SOURCE = 'nextgqa_crop'


@DATASETS.register(name='nextgqa_grounding')
class NExTGQAGroundingDataset(GroundingDataset, NExTGQADataset):

    SOURCE = 'nextgqa_grounding'
    DATA_TYPE = 'grounding'
