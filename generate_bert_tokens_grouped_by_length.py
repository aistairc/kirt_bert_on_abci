import glob
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from random import random

from tqdm import tqdm

from meta import DB
from transformers.tokenization_bert import BertTokenizer


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


def set_discontiguous_flag(tokens_length, previous_tokens_length, new_doc_flags):
    discontiguous_flag_128 = False
    discontiguous_flag_256 = False
    discontiguous_flag_512 = False
    if tokens_length < 126:
        if new_doc_flags[0] or (not new_doc_flags[0] and previous_tokens_length == 128):
            discontiguous_flag_128 = False
        else:
            discontiguous_flag_128 = True
        previous_tokens_length = 128
    elif tokens_length < 254:
        if new_doc_flags[1] or (not new_doc_flags[1] and previous_tokens_length == 256):
            discontiguous_flag_256 = False
        else:
            discontiguous_flag_256 = True
        previous_tokens_length = 256
    else:
        if new_doc_flags[2] or (not new_doc_flags[2] and previous_tokens_length == 512):
            discontiguous_flag_512 = False
        else:
            discontiguous_flag_512 = True
        previous_tokens_length = 512

    return discontiguous_flag_128, discontiguous_flag_256, discontiguous_flag_512, previous_tokens_length


def add_tokens(init_tokens, tokens_to_add, sep_token='[SEP]', discontiguous_flag=False, max_num_tokens=128):
    if len(init_tokens) > 0 and discontiguous_flag:
        init_tokens.append(sep_token)

    init_tokens.extend(tokens_to_add)
    truncate_seq_pair(init_tokens, [], max_num_tokens - 2)


def tokenize_grouped_by_length(raw_text, tokenizer, output_dir):
    pid = raw_text[0]
    raw_text_file = raw_text[1]
    token_db_128 = DB(storage_dir=output_dir, name=str(pid) + '_128', to_write=True)
    token_db_256 = DB(storage_dir=output_dir, name=str(pid) + '_256', to_write=True)
    token_db_512 = DB(storage_dir=output_dir, name=str(pid) + '_512', to_write=True)

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    with token_db_128 as docs_128:
        with token_db_256 as docs_256:
            with token_db_512 as docs_512:
                with open(raw_text_file, "r", encoding="UTF-8") as f:
                    doc_128 = []
                    doc_256 = []
                    doc_512 = []
                    previous_tokens_length = 128
                    new_doc_flags = [True, True, True]  # 0: 128, 1:256, 2: 512
                    for line in tqdm(f, desc="Process {}: Loading Dataset".format(str(pid)), unit=" lines"):
                        line = line.strip()
                        if line == "":  # End of a raw document
                            new_doc_flags = [True, True, True]
                            if len(doc_128) > 0 and doc_128[-1] != sep_token:
                                doc_128.append(sep_token)
                            elif len(doc_256) > 0 and doc_256[-1] != sep_token:
                                doc_256.append(sep_token)
                            elif len(doc_512) > 0 and doc_512[-1] != sep_token:
                                doc_512.append(sep_token)
                        else:
                            tokens = tokenizer.tokenize(line)
                            discontiguous_flag_128, discontiguous_flag_256, discontiguous_flag_512, previous_tokens_length = set_discontiguous_flag(
                                len(tokens), previous_tokens_length, new_doc_flags)
                            if len(tokens) <= 126:
                                add_tokens(doc_128, tokens, discontiguous_flag=discontiguous_flag_128,
                                           max_num_tokens=128, sep_token=sep_token)
                                new_doc_flags[0] = False
                            elif len(tokens) <= 254:
                                add_tokens(doc_256, tokens, discontiguous_flag=discontiguous_flag_256,
                                           max_num_tokens=256, sep_token=sep_token)
                                new_doc_flags[1] = False
                            else:
                                add_tokens(doc_512, tokens, discontiguous_flag=discontiguous_flag_512,
                                           max_num_tokens=512, sep_token=sep_token)
                                new_doc_flags[2] = False

                        if len(doc_128) == 126:
                            doc_128 = [cls_token] + doc_128 + [sep_token]
                            docs_128.add(doc_128)
                            doc_128 = []
                        elif len(doc_256) == 254:
                            doc_256 = [cls_token] + doc_256 + [sep_token]
                            docs_256.add(doc_256)
                            doc_256 = []
                        elif len(doc_512) == 510:
                            doc_512 = [cls_token] + doc_512 + [sep_token]
                            docs_512.add(doc_512)
                            doc_512 = []

                if len(docs_128) <= 1 and len(docs_256) <= 1 and len(docs_512) <= 1:
                    # exit(
                    import warnings
                    warnings.warn(
                        "ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                        "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                        "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                        "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                        "sections or paragraphs.")


def all_to_one_db(des_db, sources_list):
    with des_db as des:
        for source_db in sources_list:
            with source_db as source:
                keys = list(source_db.shelf.keys())
                for key in keys:
                    des.add(source[key])


def merge_tokens_db(no_partition, output_dir):
    token_dbs_128 = []
    token_dbs_256 = []
    token_dbs_512 = []
    for pid in range(no_partition):
        token_dbs_128.append(DB(storage_dir=output_dir, name=str(pid) + '_128', to_write=False))
        token_dbs_256.append(DB(storage_dir=output_dir, name=str(pid) + '_256', to_write=False))
        token_dbs_512.append(DB(storage_dir=output_dir, name=str(pid) + '_512', to_write=False))

    token_db_128 = DB(storage_dir=output_dir, name='128', to_write=True)
    token_db_256 = DB(storage_dir=output_dir, name='256', to_write=True)
    token_db_512 = DB(storage_dir=output_dir, name='512', to_write=True)

    for des_db, sources_list in zip([token_db_128, token_db_256, token_db_512],
                                    [token_dbs_128, token_dbs_256, token_dbs_512]):
        all_to_one_db(des_db, sources_list)


def main():
    parser = ArgumentParser()
    parser.add_argument('--raw_text_dir', type=str, required=True, help="Raw text file used for training the BERT")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory that contains tokens shelf.db files")
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory that contains vocab.txt")
    parser.add_argument("--do_lower_case", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=args.do_lower_case)

    raw_text_files = glob.glob(os.path.join(args.raw_text_dir, "*.txt"))

    workers = Pool(len(raw_text_files))
    arguments = [((pid, raw_text_file), tokenizer, args.output_dir) for pid, raw_text_file in enumerate(raw_text_files)]
    workers.starmap(tokenize_grouped_by_length, arguments)

    # merge_tokens_db(len(raw_text_files), args.output_dir)


if __name__ == '__main__':
    main()
    # args = ['--raw_text_dir=data/splitted_raw_text/',
    #         '--output_dir=data/generated/tokens_gbl/',
    #         '--vocab_dir=data/generated/vocab_50k/']
    # main(args)
