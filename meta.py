import json
import logging
import shelve
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")
InputFeatures = namedtuple("InputFeatures", "input_ids lm_label_ids")
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class DB:
    def __init__(self, storage_dir, name='', key_prefix=None, to_write=False):

        self.storage_dir = Path(storage_dir)
        self.shelf_filepath = self.storage_dir / (name + '_shelf.db')
        flag = 'n' if to_write else 'r'
        self.shelf = shelve.open(str(self.shelf_filepath), flag=flag, protocol=-1)

        self.current_idx = len(self.shelf.keys())
        self.key_prefix = key_prefix

    def add(self, obj):
        if not obj:
            return

        self.shelf[self.key_prefix + '_' + str(self.current_idx) if self.key_prefix else str(self.current_idx)] = obj

        self.current_idx += 1

    def __len__(self):
        return self.current_idx

    def __getitem__(self, item):
        return self.shelf[str(item)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.shelf is not None:
            self.shelf.close()


class PregeneratedDataset(Dataset):

    @staticmethod
    def convert_example_to_features(example, tokenizer, max_seq_length):
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        is_random_next = example["is_random_next"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]

        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        features = InputFeatures(input_ids=input_array,
                                 input_mask=mask_array,
                                 segment_ids=segment_array,
                                 lm_label_ids=lm_label_array,
                                 is_next=is_random_next)
        return features

    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False,
                 reduce_memory_tmp_dir=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = reduce_memory_tmp_dir
            self.working_dir = Path(self.temp_dir)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                try:
                    line = line.strip()
                    example = json.loads(line)
                    features = PregeneratedDataset.convert_example_to_features(example, tokenizer, seq_len)
                    input_ids[i] = features.input_ids
                    segment_ids[i] = features.segment_ids
                    input_masks[i] = features.input_mask
                    lm_label_ids[i] = features.lm_label_ids
                    is_nexts[i] = features.is_next
                except:
                    print('Error', line)
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))
