import logging
import os
import statistics
from glob import glob

import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from meta import DB
from opt.lamb import Lamb
from opt.radam import RAdam
from opt.ranger import Ranger
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import save_model, log_training_with_local_loss

logger = logging.getLogger(__name__)

ADAMW, RADAM, LAMB, RANGER = 'adamw', 'radam', 'lamb', 'ranger'


class TokenDS(Dataset):
    def __init__(self, token_db, ids):
        self.token_db = token_db
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.token_db[self.ids[index]]


class BatchProcessor():
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def _get_segment_ids(self, input):
        segment_ids = [0] * len(input)
        if not self.args.ignore_segment_ids:
            sep_index = input.index(self.tokenizer.sep_token) + 1
            segment_ids[sep_index:] = [1] * (len(input) - sep_index)

        assert len(segment_ids) == len(input)

        return segment_ids

    def mask_tokens(self, inputs):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
        segment_ids = torch.tensor([self._get_segment_ids(input) for input in inputs])
        inputs = torch.tensor([self.tokenizer.convert_tokens_to_ids(input) for input in inputs])

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                               labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, segment_ids, labels

    def process_batch(self, batch):
        return self.mask_tokens(batch)


class VLTrain(object):

    @staticmethod
    def train_partition(data_loader, model, optimizer, scheduler, curr_global_step, curr_losses, best_losses,
                        nb_tr_counter,
                        args,
                        desc='Iterator'):
        global_step = curr_global_step

        best_epoch_loss = best_losses[0]
        best_local_loss = best_losses[1]

        training_log_file = os.path.join(args.output_dir, "training_logs.txt")

        epoch_tr_loss = curr_losses[0]
        local_tr_losses = curr_losses[1]
        nb_tr_examples = nb_tr_counter[0]
        nb_tr_steps = nb_tr_counter[1]

        with tqdm(data_loader, desc=desc, disable=args.local_rank not in [-1, 0]) as pbar:
            for step, batch in enumerate(data_loader):
                inputs, segment_ids, labels = batch
                inputs = inputs.to(args.device)
                segment_ids = segment_ids.to(args.device)
                labels = labels.to(args.device)
                model.train()

                outputs = model(inputs, token_type_ids=segment_ids, masked_lm_labels=labels) if args.mlm else model(
                    inputs, token_type_ids=segment_ids, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    if args.fp16:
                        try:
                            from apex import amp
                        except ImportError:
                            raise ImportError(
                                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                epoch_tr_loss += loss.item()
                local_tr_losses.append(loss.item())
                if len(local_tr_losses) > args.mean_batch_loss_size:
                    del local_tr_losses[0]
                nb_tr_examples += inputs.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_epoch_loss = epoch_tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_local_loss = statistics.mean(local_tr_losses)
                pbar.set_postfix_str(
                    f"Mean epoch loss: {mean_epoch_loss:.5f} | Mean local loss: {mean_local_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.optimizer.lower() not in [RADAM, RANGER]:
                        scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                if (global_step % args.save_checkpoint_steps == 0) and \
                        (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    is_best_epoch_loss = False
                    is_best_local_loss = False
                    if mean_epoch_loss < best_epoch_loss:
                        best_epoch_loss = mean_epoch_loss
                        logging.info(
                            f"  \nNew best epoch train loss = {best_epoch_loss} at step {str(global_step)}")
                        is_best_epoch_loss = True

                    if mean_local_loss < best_local_loss:
                        best_local_loss = mean_local_loss
                        logging.info(
                            f"  \nNew best local train loss = {best_local_loss} at step {str(global_step)}")
                        is_best_local_loss = True

                    logging.info("** ** * Saving model ** ** * ")
                    save_model(model=model,
                               save_directory=args.output_dir,
                               model_name='abci_bert',
                               suffix=str(global_step) + '_' + str(mean_epoch_loss) + '_' + str(mean_local_loss),
                               )
                    log_training_with_local_loss(training_log_file, global_step, (mean_epoch_loss, mean_local_loss),
                                                 (is_best_epoch_loss, is_best_local_loss))

        curr_losses = (epoch_tr_loss, local_tr_losses)
        best_losses = (best_epoch_loss, best_local_loss)
        nb_tr_counter = (nb_tr_examples, nb_tr_steps)
        mean_losses = (mean_epoch_loss, mean_local_loss)

        return global_step, curr_losses, best_losses, nb_tr_counter, mean_losses

    @staticmethod
    def train_epoch(train_dls, model, optimizer, scheduler, epochs, curr_global_step, best_losses, args, desc=None):
        global_step = curr_global_step
        mean_losses = None
        for _ in trange(int(epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            if not best_losses:
                best_epoch_loss = float("inf")
                best_local_loss = float("inf")
            else:
                best_epoch_loss = best_losses[0]
                best_local_loss = best_losses[1]
            epoch_tr_loss = 0
            local_tr_losses = []
            nb_tr_examples, nb_tr_steps = 0, 0
            best_losses = (best_epoch_loss, best_local_loss)
            curr_losses = (epoch_tr_loss, local_tr_losses)
            nb_tr_counter = (nb_tr_examples, nb_tr_steps)
            for train_dl in train_dls:
                global_step, curr_losses, best_losses, nb_tr_counter, mean_losses = VLTrain.train_partition(train_dl,
                                                                                                            model,
                                                                                                            optimizer,
                                                                                                            scheduler,
                                                                                                            global_step,
                                                                                                            curr_losses,
                                                                                                            best_losses,
                                                                                                            nb_tr_counter,
                                                                                                            args, desc)
        return global_step, best_losses, mean_losses

    @staticmethod
    def train(model, tokenizer, args):
        """ Train the model """

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        batch_processor = BatchProcessor(tokenizer, args)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer_type = args.optimizer.lower()

        if optimizer_type == LAMB:
            optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif optimizer_type == RADAM:
            optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif optimizer_type == RANGER:
            optimizer = Ranger(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        logging.info(f"  Optimizer = {optimizer_type}")
        token_dbs = {}
        logging.info("***** Reading data *****")
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

        if tokens_dir_list:
            for tokens_dir_name, batch_size, epochs in zip(tokens_dir_list, batch_size_list, epochs_list):
                tokens_dir = os.path.join(tokens_root_dir, tokens_dir_name)
                file_names = glob(os.path.join(tokens_dir, "*.db.dat"))
                if len(file_names) == 0:
                    file_names = glob(os.path.join(tokens_dir, "*.db"))
                train_dls = []
                nb_train_examples = 0
                for file_name in file_names:
                    token_db_file_name = os.path.splitext(os.path.splitext(os.path.basename(file_name))[0])[0].replace(
                        '_shelf',
                        '')
                    token_db = DB(storage_dir=tokens_dir, name=token_db_file_name, to_write=False)
                    ids = list(token_db.shelf.keys())
                    nb_train_examples += len(token_db)
                    train_data = TokenDS(token_db, ids)
                    if args.local_rank == -1:
                        sampler = RandomSampler(train_data)
                    else:
                        sampler = DistributedSampler(train_data)
                    train_dl = DataLoader(train_data, sampler=sampler, batch_size=batch_size,
                                          collate_fn=batch_processor.process_batch, num_workers=5)
                    train_dls.append(train_dl)

                total_train_examples = epochs * nb_train_examples
                num_train_optimization_steps = int(
                    total_train_examples / batch_size / args.gradient_accumulation_steps)

                if args.local_rank != -1:
                    num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

                if optimizer_type not in [RADAM, RANGER]:
                    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                                num_training_steps=num_train_optimization_steps)
                else:
                    scheduler = None

                logging.info(f"  Tokens folder: {tokens_dir}")
                logging.info(f"  Num examples = {total_train_examples}")
                logging.info("  Batch size = %d", batch_size)
                logging.info("  Num steps = %d", num_train_optimization_steps)

                token_dbs[tokens_dir_name] = (train_dls, scheduler, epochs)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        logging.info('Initialized model and optimizer')

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            logging.info('Wrapping model with DataParallel')

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
            logging.info('Finish wrapping model with DistributedDataParallel')

        global_step = 0
        best_losses = None
        mean_losses = None
        # Remove old eval file:
        training_log_file = os.path.join(args.output_dir, "training_logs.txt")
        command = 'rm ' + training_log_file
        os.system(command)

        for tokens_dir_name, (train_dls, scheduler, epochs) in token_dbs.items():
            global_step, best_losses, mean_losses = VLTrain.train_epoch(train_dls, model, optimizer,
                                                                        scheduler, epochs, global_step, best_losses,
                                                                        args,
                                                                        desc='Training {}'.format(tokens_dir_name))

        # # Save the last model
        mean_epoch_loss, mean_local_loss = mean_losses
        best_epoch_loss, best_local_loss = best_losses

        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            is_best_epoch_loss = False
            is_best_local_loss = False
            if mean_epoch_loss < best_epoch_loss:
                best_epoch_loss = mean_epoch_loss
                logging.info(f"  \nNew best epoch train loss = {best_epoch_loss} at step {str(global_step)}")
                is_best_epoch_loss = True

            if mean_local_loss < best_local_loss:
                best_local_loss = mean_local_loss
                logging.info(f"  \nNew best local train loss = {best_local_loss} at step {str(global_step)}")
                is_best_local_loss = True

            logging.info("** ** * Saving model ** ** * ")
            save_model(model=model,
                       save_directory=args.output_dir,
                       model_name='abci_bert',
                       suffix=str(global_step) + '_' + str(mean_epoch_loss) + '_' + str(mean_local_loss),
                       )
            log_training_with_local_loss(training_log_file, global_step, (mean_epoch_loss, mean_local_loss),
                                         (is_best_epoch_loss, is_best_local_loss))
