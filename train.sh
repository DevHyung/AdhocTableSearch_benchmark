export CUDA_VISIBLE_DEVICES=0
EPOCH=5
SEED=20200401
KFOLD=3
DATASET=bench
python trainer.py --data_dir ./data/${DATASET}/${KFOLD} \
  --bert_path bert-base-uncased \
  --tabert_path ./model/tabert_base_k3/model.bin \
  --gpus 1 \
  --precision 16 \
  --max_epochs ${EPOCH} \
  --do_train \
  --train_batch_size 4 \
  --valid_batch_size 4 \
  --test_batch_size 1 \
  --seed ${SEED} \
  --output_dir ./${DATASET}_FOLD${KFOLD}_E${EPOCH}_S${SEED}/ \
  --accumulate_grad_batches 16 \