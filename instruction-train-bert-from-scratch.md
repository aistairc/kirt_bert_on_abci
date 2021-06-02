# 1/ Download and install Anaconda on ABCI (run step by step):
	```
	$ cd ~
	$ mkdir download
	$ cd download
	$ wget https://repo.continuum.io/archive/Anaconda3-2020.11-Linux-x86_64.sh
	$ bash Anaconda3-2020.11-Linux-x86_64.sh
	$ cd ~
	$ source .bashrc
	```

# 2/ Create an environment name "pytorch-test" using Anaconda:
	```
	$ conda create --name pytorch-test python=3.8
	$ conda activate pytorch-test
	```

Install libs to new environment (copy and paste to command line, choose y (yes) for all):
	```
	conda install click ftfy joblib regex tqdm requests filelock termcolor pathlib
	pip install sentencepiece loguru cchardet torch tokenizers tensorboardX
	```

# 3/ Download and unzip kirt_vctrans GitHub Repo to kirt_vctrans folder.
Prepare all your raw text data into one single text file, e.g: single.txt
Train sentencePiece:
	```
	$ qrsh -g gab50226 -l rt_G.large=1 -l h_rt=12:00:00
	$ conda activate pytorch-test
	$ cd ~
	$ cd kirt_vctrans
	$ python build_vocab.py -i data/raw/single.txt -v 32000
	```
This process generates "spm.model" and "spm.vocab"
*Note: Run on node Full (large), this phrase requires > 300GB RAM
       32000 is vocab_size*

Convert spm.vocab to vocab.txt and move to vocab folder:
	```
	$ python preprocessing.py build-bert-vocab -i spm.vocab
	$ mv vocab.txt vocab/test/
	$ exit
	```
*Note: Exit node Full after generated vocab.txt*


# 4/ Grouping sentences by the lengths, move same length tokens to folders (should use node Small):
	```
	$ qrsh -g gab50226 -l rt_G.small=1 -l h_rt=12:00:00
	$ conda activate pytorch-test
	$ cd ~
	$ cd kirt_vctrans
	$ python generate_bert_tokens_grouped_by_length.py --raw_text_dir data/raw/ --vocab_dir vocab/test/ --output_dir data/generated/tokens/  --disable_tokenize_chinese_chars
	$ cd data/generated/tokens/
	$ mkdir tokens_128 tokens_256 tokens_384 tokens_512
	$ mv *_128_* tokens_128/
	$ mv *_256_* tokens_256/
	$ mv *_384_* tokens_384/
	$ mv *_512_* tokens_512/
	$ exit
	```
*Note: data/raw/ is the directory path of "single.txt"
          vocab/test/ is directory path of "vocab.txt"
          data/generated/tokens/ is the output directory path*
Exit node Small after the process.

# 5/ Create model input / output folders. Put vocab.txt, config.json into models/test/in folder.
	```
	$ cd ~
	$ cd kirt_vctrans
	$ mkdir models
	$ mkdir models/test
	$ mkdir models/test/in
	$ mkdir models/test/out
	$ cp vocab/test/vocab.txt models/test/in
	```

# 6/ Modify train_job.sh module:
	```
	$ vi train_job.sh
	```
Note: default 4 Full Node large (set number of Nodes, number of GPUs per Node if necessary)
	```
	NUM_NODES=4 #Number of Node Large. Ussually set 1~4 nodes
	NUM_GPUS_PER_NODE=4 #Number of GPU per Node
	```
Note: options for train_on_tokens.py (folders setting, training options)
	```
	ARGS="train_on_tokens.py 
	--tokens_dir=data/generated/tokens/
	--tokens_dir_list=tokens_128,tokens_256,tokens_384,tokens_512 
	--batch_size_list=32,8,8,8 
	--epochs_list=12,3,3,3 
	--bert_model=models/test/in/ 
	--model_type=bert 
	--output_dir=models/test/out/ 
	--train_batch_size=32 
	--learning_rate=1e-4 
	--optimizer=RADAM 
	--save_checkpoint_steps=5000 
	--weight_decay=0.01 
	--mlm 
	--mean_batch_loss_size=1000 
	--epochs=30
	--disable_tokenize_chinese_chars
	```

# 7/ In case train BERT on a pre-trained model:
You have to copy the pretrained model to models/test/in
	```
	$ cp {path}/abci_bert_90000_1.1572281245778246_1.1572281245778246.bin models/test/in/pytorch_model.bin
	```
And change the option in train_job.sh from 
	```
	--bert_model=models/test/in/
	```
to
	```
	--bert_pretrained_model=models/test/in/
	```

# 8/ Modify submit_train_job.sh module
	```
	$ vi submit_train_job.sh
	```
Inside of submit_train_job.sh:
	```
	CACHE_DIR=".cache" 
	rm -rf $CACHE_DIR
	mkdir $CACHE_DIR
	qsub -g gab99999 train_job.sh
	```
Note: change gab99999 to name of the disk group on ABCI that your account belongs to

# 9/ Run submit_train_job.sh to proceed train BERT:
	```
	$ bash submit_train_job.sh
	```
Your job-array 4119943.1-4:1 ("train_job.sh") has been submitted
The output models will be located in models/test/out/

List all log files:
	```
	$ ls train_job.sh.*
	train_job.sh.o4119943.1
	train_job.sh.o4119943.2
	train_job.sh.o4119943.3
	train_job.sh.o4119943.4
	```
View the real-time process of train BERT by command:
	```
	$ tail -f  train_job.sh.o4119943.1
	```