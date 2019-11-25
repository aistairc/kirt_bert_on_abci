import json
import logging
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from meta import DB, PregeneratedDataset
from opt.lamb import Lamb
from opt.radam import RAdam
from opt.ranger import Ranger
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import save_model, log_training

logger = logging.getLogger(__name__)

ADAMW, RADAM, LAMB, RANGER = 'adamw', 'radam', 'lamb', 'ranger'


class Train(object):

    @staticmethod
    def batch_to_tensors(batch, train_batch_size):
        seq_len = len(batch[0].input_ids)
        input_ids = np.zeros(shape=(train_batch_size, seq_len), dtype=np.int32)
        lm_label_ids = np.full(shape=(train_batch_size, seq_len), dtype=np.int32, fill_value=-1)
        for i, sample in enumerate(batch):
            input_ids[i] = sample.input_ids
            lm_label_ids[i] = sample.lm_label_ids
        return (torch.tensor(input_ids.astype(np.int64)),
                torch.tensor(lm_label_ids.astype(np.int64)))

    @staticmethod
    def train(args, model, tokenizer):
        """ Train the model """

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

        samples_per_epoch = []
        epoch_dbs = []

        if args.large_train_data:
            file_names = glob(os.path.join(args.pregenerated_data, "*.db.dat"))
            if len(file_names) == 0:
                file_names = glob(os.path.join(args.pregenerated_data, "*.db"))
            epoch_db_file_map = {}

            for file_name in file_names:
                epoch_id = os.path.splitext(os.path.basename(file_name))[0].split('_')[1]
                db_file_name = os.path.splitext(os.path.splitext(os.path.basename(file_name))[0])[0].replace('_shelf',
                                                                                                             '')
                epoch_db_file_map[epoch_id] = [] if epoch_id not in epoch_db_file_map else epoch_db_file_map[epoch_id]
                epoch_db_file_map[epoch_id].append(db_file_name)

        for i in range(args.epochs):
            data_readable = True

            if args.large_train_data:
                epoch_db_keys = []
                try:
                    for db_file_name in epoch_db_file_map[str(i)]:
                        db_file = DB(storage_dir=args.pregenerated_data, name=db_file_name, to_write=False)
                        epoch_db_keys.extend([(db_key, db_file) for db_key in list(db_file.shelf.keys())])
                    samples_per_epoch.append(len(db_file))
                    epoch_dbs.append(epoch_db_keys)
                except:
                    data_readable = False
            else:
                epoch_file = args.pregenerated_data / f"epoch_{i}.json"
                metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
                if epoch_file.is_file() and metrics_file.is_file():
                    metrics = json.loads(metrics_file.read_text())
                    samples_per_epoch.append(metrics['num_training_examples'])
                else:
                    data_readable = False

            if not data_readable:
                if i == 0:
                    exit("No training data was found!")
                print(
                    f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
                print(
                    "This script will loop over the available data, but training diversity may be negatively impacted.")
                num_data_epochs = i
                break
        else:
            num_data_epochs = args.epochs

        total_train_examples = 0
        for i in range(args.epochs):
            # The modulo takes into account the fact that we may loop over limited epochs of data
            total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

        num_train_optimization_steps = int(
            total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

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

        if optimizer_type not in [RADAM, RANGER]:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=num_train_optimization_steps)

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
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {total_train_examples}")
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        logging.info(f"  Optimizer = {optimizer_type}")
        model.train()
        best_loss = float("inf")
        training_log_file = os.path.join(args.output_dir, "training_logs.txt")
        for epoch in range(args.epochs):
            if args.large_train_data:
                epoch_db_keys = epoch_dbs[epoch]
                epoch_dataset = torch.tensor([i for i in range(len(epoch_db_keys))])

            else:
                epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data,
                                                    tokenizer=tokenizer,
                                                    num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory,
                                                    reduce_memory_tmp_dir=args.reduce_memory_tmp_dir)

            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    if args.large_train_data:
                        batch = Train.batch_to_tensors(
                            [epoch_db_keys[id][1][epoch_db_keys[id][0]] for id in batch.tolist()],
                            args.train_batch_size)
                        batch = tuple(t.to(args.device) for t in batch)
                        input_ids, lm_label_ids = batch
                    else:
                        batch = tuple(t.to(args.device) for t in batch)
                        input_ids, _, _, lm_label_ids, _ = batch
                    outputs = model(input_ids, masked_lm_labels=lm_label_ids)
                    loss = outputs[0]
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    pbar.update(1)
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        if optimizer_type not in [RADAM, RANGER]:
                            scheduler.step()  # Update learning rate schedule
                        optimizer.zero_grad()
                        global_step += 1

                    if (global_step % args.save_checkpoint_steps == 0) and \
                            (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                        is_best_loss = False
                        if mean_loss < best_loss:
                            best_loss = mean_loss
                            logging.info(f"  New best train loss = {best_loss}")
                            is_best_loss = True

                        logging.info("** ** * Saving model ** ** * ")
                        save_model(model=model,
                                   save_directory=args.output_dir,
                                   model_name='abci_bert',
                                   suffix=str(global_step) + '_' + str(loss.item()),
                                   )
                        log_training(training_log_file, global_step, mean_loss, is_best_loss)

        # Save the last model
        is_best_loss = False
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            if mean_loss < best_loss:
                best_loss = mean_loss
                logging.info(f"  New best train loss = {best_loss}")
                is_best_loss = True

            logging.info("** ** * Saving model ** ** * ")
            save_model(model=model,
                       save_directory=args.output_dir,
                       model_name='abci_bert',
                       suffix=str(global_step) + '_' + str(loss.item()),
                       )
            log_training(training_log_file, global_step, mean_loss, is_best_loss)
