# ABCI-BERT
Training BERT on ABCI

# Build vocab:
Step 1: Prepare all your raw text data into one single text file, e.g: single.txt
```
python build_vocab.py -i {path}/single.txt -v {vocab_size}
```
*Note: Run on node Full, this phrase requires > 300GB RAM

Step 2: Step 1 will result two files "spm.model" and "spm.vocab" in the same directory with "build_vocab.py"

```
python preprocessing.py build-bert-vocab -i spm.vocab
```
Step 2 will result "vocab.txt" file that we can use for training the BERT. Manually move spm.vocab and spm.model file since they're not used anymore

# Pre-generate training data
pregenerate training data is a well applicable step when training BERT from large data. By this way, it help separate the need for computation resources. While data pre-processing can be done using only CPU, then training from these processed data may be performed on more powerful calculation unit like GPU or TPU. More over, data once being pre-generated can be used for different training scenarios. Pre-generating training data contains two steps:
* [Generate BERT tokens](#generate_bert_tokens)
* [Generate training data from tokens](#generate_training_data_from_tokens)

## <a name="generate_bert_tokens"></a>  Generate BERT tokens
Generate sub-word tokens from text, using BertTokenizer, and save tokens into _shelf.db binary file. 
```
python generate_bert_tokens.py --raw_text_file=data/raw_text/single.txt --vocab_dir=data/generated/vocab_50k/vocab.txt --output_dir=data/generated/tokens/ --max_seq_len=512
```

## <a name="generate_training_data_from_tokens"></a> Generate training data from tokens
Generate epoch data for training, from generated tokens, the --max_seq_len argument should be matched with generated tokens. Generated epoch data will be saved into shelf.db binary file
```
python generate_training_data_from_tokens.py --tokens_dir=data/generated/tokens/ --vocab_dir=data/generated/vocab_50k/vocab.txt --output_dir=data/generated/epochs/ --epochs_to_generate=5 --max_seq_len=512 --do_whole_word_mask --num_workers=5
```

# Training the BERT
We train the BERT following the strategy of [RoBERTa](#https://arxiv.org/pdf/1907.11692.pdf), with making use of BertForMaskedLM from [huggingface's transformers](#https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py)
```
python train_on_pregenerated.py --pregenerated_data=data/generated/epochs/ --bert_model=data/generated/abci_bert_base/ --output_dir=data/generated/abci_bert_base/model/ --train_batch_size=8 --epochs=5 --learning_rate=1e-4 --optimizer=RADAM --large_train_data --fp16 --save_checkpoint_steps=5

```
## Traing the BERT on ABCI
We also prepare a script for training the BERT on the ABCI: train.sh, where the training can be performed using multiple GPUs distributed through multiple ABCI nodes. Customize below parameters up to your usage:
```bash
#$ -l rt_G.large=1: rt_G.large = the ABCI node type (https://abci.ai/en/about_abci/cloud_service.html)
#$ -l h_rt=72:00:00: 72 = running time (hours)
NUM_NODES=2: 2 = number of ABCI nodes
#$ -t 1-2: In case of changing the NUM_NODES to X, please change this configuration to '1-X'
```
***IMPORTANT: Manually create/clear cache dir before training, while cache dir is configured in train.sh here: ***
```
CACHE_DIR=".cache"
```

# Evaluate trained BERT model
For assessing the performance of our pre-trained BERT model, we conducted the experiment on the Named Entity Recognition task using the model implemented [here](https://github.com/dnanhkhoa/ProtoNLP). This model is inspired by the paper "[Deep Exhaustive Model for Nested Named Entity Recognition](https://www.aclweb.org/anthology/D18-1309.pdf)" with replacing the LSTM layer by BERT. 
The table below shows the overall score on the development set of [Cancer Genetics 2013](http://2013.bionlp-st.org/tasks/cancer-genetics) corpus.

|  Model  | Precision | Recall |   F1  |
|:-------:|:---------:|:------:|:-----:|
| BERT-on-ABCI |     80.66 |  81.22 | 80.94 |
