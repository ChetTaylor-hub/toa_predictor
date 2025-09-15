# 3. 训练模型
echo "3. 开始训练 LSTM 模型..."
python train.py --config configs/unified_config.yaml --data_path data/test_cir_data.csv --experiment_name toa_lstm_3

echo "训练完成！"