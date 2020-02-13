# ABCI-BERT_BETA
Training [BERT](https://www.aclweb.org/anthology/N19-1423/) on ABCI (Beta version) using [Huggingface's Transformers](https://github.com/huggingface/transformers).

# *** _New updates_ ***
## Group samples by length
- In this version, we group the samples into three chunks with fix length for each chunk (128, 256, 512). Chunk with length of N contains the concatenation of sentences with length <= (N-2)
- With continuous short sentences, we consider them as a long sentence and merge them directly.
- With discontinuous short sentences, we add a separator character between them. The sentences from different documents will have separator as well
- One sample in chunk N will be built by concatenating method as above, until they reach the length of N-2. The 2 spaces are used for CLS and SEP token

## Training directly from tokens
The pipeline in alpha version is: Generate tokens -> Generate training data from tokens -> Train from generated data. In this beta version, we shorten this pipeline to: Generate tokens (grouped by length) -> Train from tokens

## Support multiprocessing when generating tokens
We realize that it's not necessary to put all raw text into one single file because it eliminates the use of multiprocessing. In this version, we recommend the user split their raw text into small files. Our codes will support process these files simultaneously (one process for each file), so we can speed up tokens generating process

## Support training from multiple tokens files
Once we finish generating tokens from multiple small files, we will have the corresponding number of tokens files. We don't have to merge them into one file, just put them all in one folder and train. 

## Support training from multiple folders
In this version, we also support training from multiple folders. With the assumption that we may need to train the BERT with samples length of 128 first, and then 256 and 512. Please note that we only support the 2-level directory, it means we have to put all the training data to the second-level subfolders under the root folder, which we defined by parameter --tokens_dir when training. E.g: 
```
├── root_tokens_dir
│   ├── tokens_128
│        ├── training tokens files ...
│   ├── tokens_256
│        ├── training tokens files ...
│   ├── tokens_512
|        ├── training tokens files ...
```
When training, each tokens subfolder will be trained sequentially with a particular batch-size and number of epochs as well. The list of tokens subfolders needs to be defined by parameter --tokens_dir_list, while the list of batch-sizes and epochs correspond to --batch_size_list and --epochs_list. If we don't define the  --batch_size_list and --epochs_list, all tokens subfolder will be trained with the same batch-size and number of epochs, which are defined by the parameters --training_batch_size and --epochs. An example of configurations is below: 
```
--tokens_dir_list=tokens_128,tokens_256,tokens_512
--batch_size_list=32,16,8
--epochs_list=2,4,6
```
We also support a special mode called various length training. Activate this mode by adding parameter --train_various_length. It's pretty the same with training with multiple folders. The only difference is that we only use one Optimizer/Scheduler during the training process, instead of each Optimizer/Scheduler for each tokens subfolder. 

## Support transfer learning from pre-trained BERT models
In this version, we also support transfer learning from pre-trained BERT models. Please note that, in case of transfer learning, we don't have to build own vocab. We're gonna use the vocab as well as the configuration file of the pre-trained BERT. Configure the parameter --bert_pretrained_model instead of --bert_model

# Prepare raw text data for training BERT
The raw text data needs to be separated each sentence per line and each document needs to be separated by one empty line. It should be better to put a white space between the last punctuation and the last word of the sentence.  

# Build vocab
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

# Generate BERT tokens
Generate sub-word tokens from text, using BertTokenizer, and save tokens into _shelf.db binary file. 
```
python generate_bert_tokens_grouped_by_length.py --raw_text_dir=data/splitted_raw_text/ --vocab_dir=data/generated/vocab_50k/vocab.txt --output_dir=data/generated/tokens/
```

# Training the BERT
We train the BERT following the strategy of [RoBERTa](#https://arxiv.org/pdf/1907.11692.pdf), with making use of BertForMaskedLM from [huggingface's transformers](#https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py)
```
python train_on_tokens.py 
   --tokens_dir=data/generated/tokens
   --tokens_dir_list=tokens_128,tokens_256,tokens_512
   --batch_size_list=32,16,8
   --batch_size_list=2,4,6
   --bert_model=models/abci_bert_base/ 
   --output_dir=data/generated/abci_bert_base/model/ 
   --train_batch_size=32 
   --epochs=2 
   --learning_rate=1e-4 
   --optimizer=RADAM  k
   --fp16 
   --save_checkpoint_steps=10000
   --mlm
   --mean_batch_loss_size=1000
   --train_various_length
```

# Traing the BERT on ABCI
## Prepare train job script
We prepare a sample script for training the BERT on the ABCI: train_job.sh, where the training can be performed using multiple GPUs distributed through multiple ABCI nodes. Customize below parameters up to your usage:
```bash
#$ -l rt_G.large=1: rt_G.large = the ABCI node type (https://abci.ai/en/about_abci/cloud_service.html)
#$ -l h_rt=72:00:00: 72 = running time (hours)
NUM_NODES=2: 2 = number of ABCI nodes
#$ -t 1-2: In case of changing the NUM_NODES to X, please change this configuration to '1-X'
```

## Submit train job
Submit your train job script using submit_train_job.sh. Customize below parameters up to your usage:
```
CACHE_DIR=".cache"
qsub -g {group_id} submit_train_job.sh
```

# Evaluate trained BERT model
For assessing the performance of our pre-trained BERT model, we conducted the experiment on the Named Entity Recognition task using the model implemented [here](https://github.com/dnanhkhoa/ProtoNLP). This model is inspired by the paper "[Deep Exhaustive Model for Nested Named Entity Recognition](https://www.aclweb.org/anthology/D18-1309.pdf)" with replacing the LSTM layer by BERT. 
The table below shows the overall score on the development set of [Cancer Genetics 2013](http://2013.bionlp-st.org/tasks/cancer-genetics) corpus. We will add more results later. 

|  Model  | Precision | Recall |   F1  |
|:-------:|:---------:|:------:|:-----:|
| BERT-on-ABCI |     82.44 |  82.48 | 83.50 |
| SciBERT | 82.36 | 82.84 | 82.60 |
| BioBERT | 82.68 | 83.70 | 83.19 |
| BERT | 79.98 | 78.70 | 79.33 |

# References

- Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186. 2019.

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov, "Robustly Optimized BERT Pretraining Approach." ArXiv. 2019.

- Mohammad Golam Sohrab and Makoto Miwa. "Deep exhaustive model for nested named entity recognition." In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2843-2849. 2018.

# Acknowledgement

The results are obtained from "Strategic Advancement of Multi-Purpose Ultra-Human Robot and Artificial Intelligence Technologies（SamuRAI） Project" and "Ultra High-Throughput Design and Prototyping Technology for Ultra Advanced Materials Development Project" commissioned by the New Energy and Industrial Technology Development Organization (NEDO) and a project commissioned by Public/Private R&D Investment Strategic Expansion PrograM (PRISM).
