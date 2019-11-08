from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from meta import DB
from transformers.tokenization_bert import BertTokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument('--raw_text_file', type=Path, required=True, help="Raw text file used for training the BERT")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory that contains tokens shelf.db files")
    parser.add_argument("--vocab_dir", type=str, required=True, help="Directory that contains vocab.txt")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=-1)

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=args.do_lower_case)
    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_len = min(args.max_seq_len, tokenizer.max_len_single_sentence)

    with DB(storage_dir=args.output_dir, to_write=True) as docs:
        with args.raw_text_file.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            # exit(
            import warnings
            warnings.warn(
                "ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                "sections or paragraphs.")


if __name__ == '__main__':
    main()
