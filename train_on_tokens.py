import logging
import os
import random
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from trainer import Train
from transformers import BertConfig, CONFIG_NAME
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert import BertTokenizer
from various_length_trainer import VLTrain

InputFeatures = namedtuple("InputFeatures", "input_ids lm_label_ids")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = ArgumentParser()
    parser.add_argument('--tokens_dir', type=Path, required=True,
                        help="Directory that contains tokens data")
    parser.add_argument('--tokens_dir_list', type=str, default=None,
                        help="List token sub folder int tokens_dir, separated by ',', used for various length training")
    parser.add_argument('--batch_size_list', type=str, default=None,
                        help="List batch size corresponding to tokens_dir_list, separated by ',', "
                             "used for various length training")
    parser.add_argument('--epochs_list', type=str, default=None,
                        help="List number of epochs corresponding to tokens_dir_list, separated by ',', "
                             "used for various length training")

    parser.add_argument('--bert_pretrained_model', type=str, default=None,
                        help="Path / shortcut name to pre-trained model or shortcut name selected in the list, "
                             "used for transfer-learning")
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str,
                        help="Directory that has to contains the files vocab.txt and config.json")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--large_train_data", action="store_true",
                        help="In case of training with very large data that can't use np.memmap")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Optimizer.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--save_checkpoint_steps",
                        type=int,
                        default=10000,
                        help="Save checkpoints every this many steps.")
    parser.add_argument("--optimizer",
                        type=str,
                        default='ADAM',
                        help="Optimizer to use. (ADAM, LAMB, RADAM)")
    parser.add_argument("--opt_level",
                        type=str,
                        default='O1',
                        help="opt_level offered by apex, using when training with fp16 configuration.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--mean_batch_loss_size", type=int, default=1000,
                        help="Mean batch loss is counted based on K nearest batch. This param defines K")
    parser.add_argument("--train_various_length", action='store_true',
                        help="Train with various length.")
    parser.add_argument("--ignore_segment_ids", action='store_true',
                        help="Take into account segment_ids or not.")

    args = parser.parse_args()

    assert args.tokens_dir.is_dir(), \
        "--tokens_dir should point to the folder of files made by generate_bert_tokens_grouped_by_length.py!"

    assert args.bert_pretrained_model or args.bert_model, "at least --bert_pretrained_model or --bert_model is required"

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Set seed
    set_seed(args)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if not args.bert_pretrained_model:
        config = BertConfig.from_json_file(
            os.path.join(args.bert_model, CONFIG_NAME)
        )
        model = BertForMaskedLM(config)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    else:
        model = BertForMaskedLM.from_pretrained(args.bert_pretrained_model)
        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_model, do_lower_case=args.do_lower_case)

    model.to(args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training parameters %s", args)

    best_losses = None
    global_step = 0

    if not args.train_various_length:
        if args.tokens_dir_list:
            tokens_root_dir = args.tokens_dir
            tokens_dir_list = args.tokens_dir_list.split(',')
            if not args.batch_size_list:
                batch_size_list = [args.train_batch_size] * len(tokens_dir_list)
            else:
                batch_size_list = args.batch_size_list.split(',')
                batch_size_list = [int(batch_size) for batch_size in batch_size_list]

            if not args.epochs_list:
                epochs_list = [args.epochs] * len(tokens_dir_list)
            else:
                epochs_list = args.epochs_list.split(',')
                epochs_list = [int(epochs) for epochs in epochs_list]

            for tokens_dir, batch_size, epochs in zip(tokens_dir_list, batch_size_list, epochs_list):
                args.tokens_dir = os.path.join(tokens_root_dir, tokens_dir)
                args.train_batch_size = batch_size
                args.epochs = epochs
                global_step, best_losses = Train.train(model=model, tokenizer=tokenizer, args=args,
                                                       best_losses=best_losses,
                                                       curr_global_step=global_step)

        else:
            Train.train(model=model, tokenizer=tokenizer, args=args)
    else:
        VLTrain.train(model=model, tokenizer=tokenizer, args=args)


if __name__ == '__main__':
    main()
