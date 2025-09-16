#!/bin/bash

# 训练参数
DATA_PATH="data/weather.csv"      # 请根据实际路径修改
EPOCHS=20
BATCH_SIZE=32
SEQ_LEN=96
PRED_LEN=24
D_MODEL=64
N_HEADS=4
D_FF=128
LAYERS=2
DROPOUT=0.1

# 训练 iTransformer
echo "Training iTransformer on weather dataset..."
python train.py \
  --model itransformer \
  --data $DATA_PATH \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --d_ff $D_FF \
  --e_layers $LAYERS \
  --dropout $DROPOUT \
  --save_path "itransformer_weather.pth"

# 训练 AdapTFormer
echo "Training AdapTFormer on weather dataset..."
python train.py \
  --model adaptformer \
  --data $DATA_PATH \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --d_ff $D_FF \
  --e_layers $LAYERS \
  --dropout $DROPOUT \
  --save_path "adaptformer_weather.pth"

echo "训练完成，权重已保存为："
echo "  - itransformer_weather.pth"
echo "  - adaptformer_weather.pth"