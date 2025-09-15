#!/bin/bash

# PyTorch 序列预测项目快速开始脚本

echo "=== PyTorch 序列预测项目快速开始 ==="

# 1. 安装依赖
echo "1. 安装项目依赖..."
pip install -r requirements.txt

# 2. 生成示例数据
echo "2. 生成CIR-TOA示例数据..."
python generate_cir_toa_data.py

# 3. 训练模型
echo "3. 开始训练 LSTM 模型..."
python train.py --config configs/unified_config.yaml --data_path data/test_cir_data.csv --experiment_name cir_toa_lstm

echo "训练完成！"

# 4. 评估模型
echo "4. 评估模型..."
python evaluate.py 
    --experiment_dir results/cir_toa_lstm 
    --data_path data/test_cir_data.csv 
    --use_best_model

# 5. 推理示例
echo "5. 推理示例..."
python inference.py 
    --experiment_dir results/cir_toa_lstm 
    --input_path data/test_cir_data.csv 
    --output_path results/cir_toa_lstm/predictions.csv 
    --use_best_model

echo "=== 快速开始完成！==="
echo "检查以下文件："
echo "- 实验目录: results/cir_toa_lstm/"
echo "- 训练日志: results/cir_toa_lstm/*.log"
echo "- 最佳模型: results/cir_toa_lstm/best_model.pth"
echo "- 最终模型: results/cir_toa_lstm/final_model.pth"
echo "- 预处理器: results/cir_toa_lstm/preprocessor.pkl"
echo "- 训练配置: results/cir_toa_lstm/config.yaml"
echo "- 训练结果: results/cir_toa_lstm/results.yaml"
echo "- 检查点: results/cir_toa_lstm/checkpoints/"
echo "- 预测结果: results/cir_toa_lstm/predictions.csv"
