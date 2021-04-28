export CUDA_VISIBLE_DEVICES=0
ckpt_file=./bench_FOLD1_E1_S20200401/checkpoint-epoch=02.ckpt
SEED=20200401
KFOLD=1
DATASET=bench
echo ${ckpt_file}
python predict.py --data_dir ../data/${DATASET}/${KFOLD}/ \
  --bert_path bert-base-uncased \
  --tabert_path ../model/tabert_base_k3/model.bin \
  --gpus 1 \
  --test_batch_size 1 \
  --ckpt_file ${ckpt_file} \
  --output_file ${ckpt_file}.csv \
  --qrel_file ../data/${DATASET}/${KFOLD}/test.jsonl.qrels \
  --result_file ${ckpt_file}.result \











