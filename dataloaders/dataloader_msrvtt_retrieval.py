from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.data_utils import mask_tokens, mask_video_frames


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
        self,
        csv_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
    ):
        self.data = pd.read_csv(csv_path)
        self.feature_dict = pickle.load(open(features_path, "rb"))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.feature_size = self.feature_dict[self.data["video_id"].values[0]].shape[-1]

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Masked Language Modeling (MLM)
            masked_tokens, token_labels = mask_tokens(words, self.tokenizer)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_masked_text,
            pairs_token_labels,
            choice_video_ids,
        )

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros(
            (len(choice_video_ids), self.max_frames, self.feature_size),
            dtype=np.float64,
        )
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[: self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = (
                max_video_length[i]
                if max_video_length[i] > slice_shape[0]
                else slice_shape[0]
            )
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][: slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Masked Frame Modeling (MFM)
        masked_video, video_labels_index = mask_video_frames(video, max_video_length)

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        video_id = self.data["video_id"].values[idx]
        sentence = self.data["sentence"].values[idx]

        (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_masked_text,
            pairs_token_labels,
            choice_video_ids,
        ) = self._get_text(video_id, sentence)

        video, video_mask, masked_video, video_labels_index = self._get_video(
            choice_video_ids
        )

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            video,
            video_mask,
            pairs_masked_text,
            pairs_token_labels,
            masked_video,
            video_labels_index,
        )


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""

    def __init__(
        self,
        csv_path,
        json_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        unfold_sentences=False,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, "r"))
        self.feature_dict = pickle.load(open(features_path, "rb"))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.feature_size = self.feature_dict[self.csv["video_id"].values[0]].shape[-1]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv["video_id"].values)
            self.sentences_dict = {}
            for itm in self.data["sentences"]:
                if itm["video_id"] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (
                        itm["video_id"],
                        itm["caption"],
                    )
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data["sentences"]:
                self.sentences[itm["video_id"]].append(itm["caption"])
                num_sentences += 1
                s_video_id_set.add(itm["video_id"])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data["videos"]:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Masked Language Modeling (MLM)
            masked_tokens, token_labels = mask_tokens(words, self.tokenizer)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_masked_text,
            pairs_token_labels,
            choice_video_ids,
        )

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros(
            (len(choice_video_ids), self.max_frames, self.feature_size),
            dtype=np.float64,
        )
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[: self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = (
                max_video_length[i]
                if max_video_length[i] > slice_shape[0]
                else slice_shape[0]
            )
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][: slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Masked Frame Modeling (MFM)
        masked_video, video_labels_index = mask_video_frames(video, max_video_length)

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv["video_id"].values[idx], None
        (
            pairs_text,
            pairs_mask,
            pairs_segment,
            pairs_masked_text,
            pairs_token_labels,
            choice_video_ids,
        ) = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(
            choice_video_ids
        )

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            video,
            video_mask,
            pairs_masked_text,
            pairs_token_labels,
            masked_video,
            video_labels_index,
        )
