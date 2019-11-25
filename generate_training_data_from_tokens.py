import collections
import logging
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from random import random, shuffle, choice

import numpy as np
from tqdm import tqdm

from meta import DB
from meta import InputFeatures
from transformers.tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def convert_sample_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             lm_label_ids=lm_label_array)
    return features


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def create_samples_from_document(doc_idx, doc_database, args, tokenizer, vocab_list, last_chunk):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    # get config parameters
    max_seq_length = args.max_seq_len
    short_seq_prob = args.short_seq_prob
    masked_lm_prob = args.masked_lm_prob
    separate_sentences = args.separate_sentences
    max_predictions_per_seq = args.max_predictions_per_seq
    whole_word_mask = args.do_whole_word_mask

    document = doc_database[doc_idx]
    # Account for [CLS], ..., [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens

    # We concatenate all of the tokens from one or more documents into a long sequence.
    # An option for adding or not [SEP] to every end of sentence
    # [SEP] must be added to every end of document
    samples = []
    # current_chunk = []
    # current_length = 0
    current_chunk = last_chunk[0]
    current_length = last_chunk[1]
    if not separate_sentences and current_length > 0:
        # Add [SEP] token to the end of last document
        current_chunk.append(["[SEP]"])
        current_length += 1
        target_seq_length -= 1
    i = 0
    try:
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if separate_sentences:
                current_length += 1  # for [SEP]
                target_seq_length -= 1
            # if i == len(document) - 1 or current_length >= target_seq_length:
            if current_length >= target_seq_length:
                # Packaging and add to samples
                if current_chunk:
                    tokens = []
                    for j in range(len(current_chunk)):
                        tokens.extend(current_chunk[j])
                        if separate_sentences:
                            tokens.extend(["[SEP]"])
                    truncate_seq_pair(tokens, [], max_num_tokens)

                    tokens = ["[CLS]"] + tokens + ["[SEP]"]

                    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                    sample = {
                        "tokens": tokens,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_labels": masked_lm_labels}
                    features = convert_sample_to_features(sample, tokenizer, max_seq_length)
                    samples.append(features)
                    current_chunk = []
                    current_length = 0

            i += 1

        return samples, (current_chunk, current_length)
    except Exception:
        traceback.print_exc()


def create_training_samples(docs, doc_keys, tokenizer, vocab_list, args, epoch, storage_dir):
    with DB(name='epoch_' + str(epoch), storage_dir=storage_dir, to_write=True) as samples_db:
        last_chunk = ([], 0)
        for doc_key in tqdm(doc_keys, desc="Epoch_" + str(epoch)):
            samples, last_chunk = create_samples_from_document(doc_key, docs, args, tokenizer, vocab_list, last_chunk)
            for sample in samples:
                samples_db.add(sample)


def main():
    parser = ArgumentParser()
    parser.add_argument('--tokens_dir', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory that contains vocab.txt")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--separate_sentences", action="store_true")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=-1)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=args.do_lower_case)
    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_len = min(args.max_seq_len, tokenizer.max_len_single_sentence)

    vocab_list = list(tokenizer.vocab.keys())

    with DB(storage_dir=args.tokens_dir, to_write=False) as docs:
        doc_keys = list(docs.shelf.keys())
        logger.info('Num docs: %d', len(doc_keys))

        workers = Pool(min(args.num_workers, args.epochs_to_generate))
        arguments = [(docs, doc_keys, tokenizer, vocab_list, args, epoch, args.output_dir) for epoch in
                     range(args.epochs_to_generate)]
        workers.starmap(create_training_samples, arguments)


if __name__ == '__main__':
    main()
