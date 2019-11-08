# -*- coding: utf-8 -*-
import os.path
import re
from random import shuffle

import click
import pubmed_parser as pp
from loguru import logger

from ssplit import regex_sentence_boundary_gen
from transformers import BasicTokenizer
from utils import *

MIN_TOKEN_COUNT = 5
MIN_SENTENCE_COUNT = 2
MIN_WORD_TOKENS_RATIO = 0.40  # Percentage of words in the tokens
MIN_LETTER_CHAR_RATIO = 0.60  # Percentage of letters in the string

basic_tokenizer = BasicTokenizer(
    do_lower_case=False, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
)


@click.group()
def cli():
    pass


def split_sentences(doc):
    for start, end in regex_sentence_boundary_gen(doc):
        yield doc[start:end].strip()


def separate_punctuations(sentence):
    return " ".join(basic_tokenizer.tokenize(sentence)).strip()


def add_punctuation(sentence):
    sentence = sentence.strip()
    if re.search(r"[^.?!]$", sentence):
        sentence += "."
    return sentence


def is_sentence(string):
    # Checks if the string is an English sentence
    # Adapted from https://github.com/allenai/scibert

    tokens = string.split()
    num_tokens = len(tokens)

    # Minimum number of words per sentence
    if num_tokens < MIN_TOKEN_COUNT:
        return False

    # Most tokens should be words
    if sum([token.isalpha() for token in tokens]) / num_tokens < MIN_WORD_TOKENS_RATIO:
        return False

    # Most characters should be letters, not numbers and not special characters
    if sum([c.isalpha() for c in string]) / len(string) < MIN_LETTER_CHAR_RATIO:
        return False

    return True


def is_doc(doc):
    assert isinstance(doc, list)
    return sum(1 for sentence in doc if is_sentence(sentence)) >= MIN_SENTENCE_COUNT


def remove_adjacent_duplicates(doc):
    sentences = []
    previous_sentence = None
    for sentence in doc:
        if previous_sentence != sentence:
            previous_sentence = sentence
            sentences.append(sentence)
    return sentences


@cli.command()
@click.option("-i", "--sentencepiece_vocab_file", required=True, type=click.Path())
def build_BERT_vocab(sentencepiece_vocab_file):
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[unused1]",
        "[unused2]",
        "[unused3]",
        "[unused4]",
        "[unused5]",
        "[unused6]",
        "[unused7]",
        "[unused8]",
        "[unused9]",
        "[unused10]",
        "[unused11]",
        "[unused12]",
        "[unused13]",
        "[unused14]",
        "[unused15]",
        "[unused16]",
        "[unused17]",
        "[unused18]",
        "[unused19]",
        "[unused20]",
        "[unused21]",
        "[unused22]",
        "[unused23]",
        "[unused24]",
        "[unused25]",
        "[unused26]",
        "[unused27]",
        "[unused28]",
        "[unused29]",
        "[unused30]",
        "[unused31]",
        "[unused32]",
        "[unused33]",
        "[unused34]",
        "[unused35]",
        "[unused36]",
        "[unused37]",
        "[unused38]",
        "[unused39]",
        "[unused40]",
        "[unused41]",
        "[unused42]",
        "[unused43]",
        "[unused44]",
        "[unused45]",
        "[unused46]",
        "[unused47]",
        "[unused48]",
        "[unused49]",
        "[unused50]",
        "[unused51]",
        "[unused52]",
        "[unused53]",
        "[unused54]",
        "[unused55]",
        "[unused56]",
        "[unused57]",
        "[unused58]",
        "[unused59]",
        "[unused60]",
        "[unused61]",
        "[unused62]",
        "[unused63]",
        "[unused64]",
        "[unused65]",
        "[unused66]",
        "[unused67]",
        "[unused68]",
        "[unused69]",
        "[unused70]",
        "[unused71]",
        "[unused72]",
        "[unused73]",
        "[unused74]",
        "[unused75]",
        "[unused76]",
        "[unused77]",
        "[unused78]",
        "[unused79]",
        "[unused80]",
        "[unused81]",
        "[unused82]",
        "[unused83]",
        "[unused84]",
        "[unused85]",
        "[unused86]",
        "[unused87]",
        "[unused88]",
        "[unused89]",
        "[unused90]",
        "[unused91]",
        "[unused92]",
        "[unused93]",
        "[unused94]",
        "[unused95]",
        "[unused96]",
        "[unused97]",
        "[unused98]",
        "[unused99]",
        "[unused100]",
    ]

    exclusive_tokens = {"<unk>"}

    for line in read_lines(sentencepiece_vocab_file, encoding="UTF-8"):
        token, _ = line.split("\t")
        token = token.strip()

        if token.lower() in exclusive_tokens:
            continue

        token = (
            re.sub(r"^▁", "", token)
            if token.startswith("▁")
            else token
            if re.match(r"\[.+\]|<.+>", token)
            else "##" + token
        )

        tokens.append(token)

    write_lines(tokens,  "vocab.txt")


def handle_pubmed(corpus_file, preprocessed_file):
    try:
        article = pp.parse_pubmed_xml(corpus_file, nxml=True)
        if article:
            title = article.get("full_title")
            abstract = article.get("abstract")

            doc = []
            for sentences in map(split_sentences, [title, abstract]):
                for sentence in sentences:
                    if sentence.strip():
                        doc.append(
                            separate_punctuations(
                                add_punctuation(normalize_string(sentence))
                            )
                        )

            previous_section_header = None

            for paragraph in pp.parse_pubmed_paragraph(corpus_file, all_paragraph=True):
                section_header = paragraph.get("section").strip()

                if section_header and previous_section_header != section_header:
                    previous_section_header = section_header

                    for sentence in split_sentences(section_header):
                        if sentence.strip():
                            doc.append(
                                separate_punctuations(
                                    add_punctuation(normalize_string(sentence))
                                )
                            )

                section_content = paragraph.get("text")
                for sentence in split_sentences(section_content):
                    if sentence.strip():
                        doc.append(
                            separate_punctuations(
                                add_punctuation(normalize_string(sentence))
                            )
                        )

            doc = remove_adjacent_duplicates(doc)

            if doc and is_doc(doc):
                write_lines([json.dumps(doc, ensure_ascii=False)], preprocessed_file)
            else:
                logger.info("Skip file: {}", corpus_file)
        else:
            logger.info("No article file: {}", corpus_file)
    except Exception as ex:
        logger.exception(ex)


def handle_medline(corpus_file, preprocessed_file):
    try:
        docs = []
        for article in pp.parse_medline_xml(corpus_file):
            title = article.get("title")
            abstract = article.get("abstract")

            doc = []
            for sentences in map(split_sentences, [title, abstract]):
                for sentence in sentences:
                    if sentence.strip():
                        doc.append(
                            separate_punctuations(
                                add_punctuation(normalize_string(sentence))
                            )
                        )

            doc = remove_adjacent_duplicates(doc)

            if is_doc(doc):
                docs.append(doc)

        if docs:
            write_lines(
                (json.dumps(doc, ensure_ascii=False) for doc in docs), preprocessed_file
            )
        else:
            logger.info("Skip file: {}", corpus_file)
    except Exception as ex:
        logger.exception(ex)


@cli.command()
@click.option("-i", "--corpus_dir", required=True, type=click.Path())
@click.option("-o", "--preprocessed_corpus_dir", required=True, type=click.Path())
@click.option(
    "-f", "--handle_func", required=True, type=click.Choice(["pubmed", "medline"])
)
@click.option("-w", "--max_workers", default=20, show_default=True, type=click.INT)
def preprocess_corpus(corpus_dir, preprocessed_corpus_dir, handle_func, max_workers=20):
    if isinstance(handle_func, str):
        handle_func = {"pubmed": handle_pubmed, "medline": handle_medline}[handle_func]

    make_dirs(preprocessed_corpus_dir)

    corpus_files = glob(corpus_dir + "/**/*.*", recursive=True)

    preprocessed_corpus_files = map(
        lambda fn: norm_path(preprocessed_corpus_dir, os.path.basename(fn) + ".json"),
        corpus_files,
    )

    for _ in run_parallel(
            handle_func,
            zip(corpus_files, preprocessed_corpus_files),
            max_workers=max_workers,
    ):
        pass


@cli.command()
@click.option("-i", "--corpus_dir", required=True, type=click.Path())
@click.option("-o", "--preprocessed_corpus_file", required=True, type=click.Path())
def merge_corpus(corpus_dir, preprocessed_corpus_file):
    corpus_files = glob(corpus_dir + "/**/*.json", recursive=True)

    shuffle(corpus_files)

    mask = len(norm_path(corpus_dir)) + 1

    write_lines(
        map(lambda fn: fn[mask:], corpus_files), preprocessed_corpus_file + ".map"
    )

    def read_docs():
        add_blank_line = False
        for corpus_file in tqdm(corpus_files):
            for doc in read_lines(corpus_file, encoding="UTF-8"):
                if add_blank_line:
                    yield ""

                add_blank_line = True

                for sentence in json.loads(doc):
                    yield sentence

    write_lines(read_docs(), preprocessed_corpus_file)


@cli.command()
@click.option("-s", "--source_vocab_file", required=True, type=click.Path())
@click.option("-t", "--target_vocab_file", required=True, type=click.Path())
def compute_vocab_iou(source_vocab_file, target_vocab_file):
    source_subwords = set(read_lines(source_vocab_file, encoding="UTF-8"))
    target_subwords = set(read_lines(target_vocab_file, encoding="UTF-8"))
    # denominator = len(source_subwords | target_subwords)
    denominator = len(source_subwords)
    iou = 0.0
    if denominator:
        iou = len(source_subwords & target_subwords) / denominator

    logger.info("IoU score: {}", iou)

    return iou


@cli.command()
@click.option("-i", "--corpus_file", required=True, type=click.Path())
def statistics(corpus_file):
    num_docs = 0
    num_sentences = 0

    has_sentence = False

    for line in tqdm(read_lines(corpus_file)):
        if line.strip():
            num_sentences += 1
            has_sentence = True
        else:
            if has_sentence:
                num_docs += 1
                has_sentence = False

    if has_sentence:
        num_docs += 1

    logger.info("# Docs: {}", num_docs)
    logger.info("# Sentences: {}", num_sentences)


if __name__ == "__main__":
    cli()
