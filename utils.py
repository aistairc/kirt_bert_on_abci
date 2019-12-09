# -*- coding: utf-8 -*-
import glob as _glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cchardet
import ftfy
import joblib
import regex
import torch
from tqdm import tqdm


def normalize_string(string):
    return regex.sub(r"\s+", " ", ftfy.fix_text(string, normalization="NFC")).strip()


def norm_path(*paths):
    return os.path.relpath(os.path.normpath(os.path.join(os.getcwd(), *paths)))


def make_dirs(*paths):
    os.makedirs(norm_path(*paths), exist_ok=True)


def read_file(filename):
    with open(norm_path(filename), "rb") as f:
        return f.read()


def write_file(data, filename):
    make_dirs(os.path.dirname(filename))
    with open(norm_path(filename), "wb") as f:
        f.write(data)


def detect_file_encoding(filename):
    return cchardet.detect(read_file(filename))["encoding"]


def read_text(filename, encoding=None):
    encoding = encoding or detect_file_encoding(filename)
    with open(norm_path(filename), "r", encoding=encoding) as f:
        return f.read()


def write_text(text, filename, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(norm_path(filename), "w", encoding=encoding) as f:
        f.write(text)


def read_lines(filename, encoding=None):
    encoding = encoding or detect_file_encoding(filename)
    with open(norm_path(filename), "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(norm_path(filename), "w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)


def read_json(filename, encoding=None):
    return json.loads(read_text(filename, encoding=encoding))


def write_json(obj, filename, indent=2, encoding="UTF-8"):
    write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False), filename, encoding=encoding
    )


def deserialize(filename, mmap_mode=None):
    return joblib.load(filename=norm_path(filename), mmap_mode=mmap_mode)


def serialize(obj, filename, compress=9, protocol=None, cache_size=None):
    make_dirs(os.path.dirname(filename))
    joblib.dump(
        value=obj,
        filename=norm_path(filename),
        compress=compress,
        protocol=protocol,
        cache_size=cache_size,
    )


def glob(pathname, *, recursive=False):
    return _glob.glob(pathname=norm_path(pathname), recursive=recursive)


def run_parallel(func, args, max_workers=None, timeout=None, chunk_size=1):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = list(args)
        kwargs = {
            "total": len(args),
            "unit": " tasks",
            "unit_scale": True,
            "leave": True,
        }
        for r in tqdm(
                executor.map(func, *zip(*args), timeout=timeout, chunksize=chunk_size),
                **kwargs
        ):
            yield r


def save_model(model, save_directory, model_name='pytorch_model', suffix=None, ext='bin'):
    assert os.path.isdir(
        save_directory), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model it-self if we are using distributed training
    model_to_save = model.module if hasattr(model,
                                            'module') else model

    # Save configuration file
    model_to_save.config.save_pretrained(save_directory)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, model_name + ('_' + suffix if suffix else '') + '.' + ext)
    torch.save(model_to_save.state_dict(), output_model_file)
    # tokenizer.save_pretrained(output_dir)


def log_training(log_file, global_step, loss, is_best_loss):
    with open(log_file, "a+") as writer:
        writer.write("Global step = %s\n" % str(global_step))
        if is_best_loss:
            writer.write("Loss = %s(New best loss)\n" % str(loss))
        else:
            writer.write("Loss = %s\n" % str(loss))


def log_training_with_local_loss(log_file, global_step, loss, is_best_loss):
    mean_epoch_loss = loss[0]
    mean_local_loss = loss[1]
    is_best_epoch_loss = is_best_loss[0]
    is_best_local_loss = is_best_loss[1]
    with open(log_file, "a+") as writer:
        writer.write("Global step = %s\n" % str(global_step))
        if is_best_epoch_loss:
            writer.write("Mean epoch loss = %s(New best epoch loss)\n" % str(mean_epoch_loss))
        else:
            writer.write("Mean epoch loss = %s\n" % str(mean_epoch_loss))

        if is_best_local_loss:
            writer.write("Mean local loss = %s(New best local loss)\n" % str(mean_local_loss))
        else:
            writer.write("Mean local loss = %s\n" % str(mean_local_loss))


class TqdmStream(object):
    def write(self, buffer):
        if buffer.strip():
            tqdm.write(buffer)
