export CUDA_VISIBLE_DEVICES=0
clear
echo -e "Fold, Epoch Input: "
read KFOLD EPOCH
# 1 :  12 13 14
# 2 :  11 12 14
# 3 :  07 09 12
# 4 :
# 5 :
#KFOLD=5
ckpt_file=./bench_FOLD${KFOLD}_E10_S20200401/checkpoint-epoch=${EPOCH}.ckpt

DATASET=bench
rm -r ../data/${DATASET}/${KFOLD}/faiss/
echo ${ckpt_file}
python predict_faiss.py --data_dir ../data/${DATASET}/${KFOLD}/ \
  --bert_path bert-base-uncased \
  --index dot \
  --tabert_path ../model/tabert_base_k3/model.bin \
  --gpus 1 \
  --ckpt_file ${ckpt_file} \
  --output_file ${ckpt_file}_faiss.csv \
  --qrel_file ../data/${DATASET}/${KFOLD}/test.jsonl.qrels \
  --result_file ${ckpt_file}_faiss.result
