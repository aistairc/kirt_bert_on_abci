CACHE_DIR=".cache"
rm -rf $CACHE_DIR
mkdir $CACHE_DIR

qsub -g {group_id} submit_train_job.sh