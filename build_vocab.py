# -*- coding: utf-8 -*-
import click
import sentencepiece as spm
from loguru import logger

from utils import *


@click.command()
@click.option("-i", "--corpus", required=True, type=click.Path())
@click.option(
    "-m", "--model_prefix", default="spm", show_default=True, type=click.STRING
)
@click.option(
    "-p",
    "--model_type",
    default="bpe",
    show_default=True,
    type=click.Choice(["unigram", "bpe", "char", "word"]),
)
@click.option(
    "-v", "--vocab_size", default=32000, show_default=True, type=click.INT
)
@click.option(
    "-c",
    "--character_coverage",
    default=0.9999,
    show_default=True,
    type=click.FLOAT,
)
@click.option(
    "-n",
    "--input_sentence_size",
    default=200000000,
    show_default=True,
    type=click.INT,
)
@click.option(
    "-l",
    "--max_sentencepiece_length",
    default=32,
    show_default=True,
    type=click.INT,
)
@click.option(
    "-s",
    "--shuffle_input_sentence",
    default=True,
    show_default=True,
    type=click.BOOL,
)
@click.option(
    "-t", "--num_threads", default=128, show_default=True, type=click.INT
)
def train(
    corpus,
    model_prefix,
    model_type,
    vocab_size,
    character_coverage,
    input_sentence_size,
    max_sentencepiece_length,
    shuffle_input_sentence,
    num_threads,
):
    # https://github.com/google/sentencepiece/issues/9#issuecomment-289352218

    corpus_files = None
    if os.path.isfile(corpus):
        corpus_files = [corpus]
    elif os.path.isdir(corpus):
        corpus_files = glob(corpus + "/**/*", recursive=True)

    assert corpus_files, "Corpus not found."

    corpus_file = ",".join(corpus_files)

    args = (
        "--input={} --model_prefix={} "
        "--model_type={} --vocab_size={} --character_coverage={} "
        "--bos_id=-1 --eos_id=-1 --pad_id=-1 --unk_id=0 "
        "--input_sentence_size={} --max_sentencepiece_length={} "
        "--shuffle_input_sentence={} --num_threads={}".format(
            corpus_file,
            model_prefix,
            model_type,
            vocab_size,
            character_coverage,
            input_sentence_size,
            max_sentencepiece_length,
            shuffle_input_sentence,
            num_threads,
        )
    )

    logger.info(args)

    spm.SentencePieceTrainer.Train(args)


if __name__ == "__main__":
    train()
