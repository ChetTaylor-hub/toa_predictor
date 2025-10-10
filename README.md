# PyTorch CIR-TOA序列预测模型训练项目

这是一个用于处理CIR（Channel Impulse Response）序列数据并预测TOA（Time of Arrival）的PyTorch项目框架。

## 项目特点

- **输入**: CIR序列数据（信道脉冲响应）
- **输出**: TOA整数值（到达时间）
- **模型**: 支持LSTM和Transformer架构
- **框架**: 高度模块化和可扩展的设计

## 项目结构

```
├── configs/                 # 配置文件
│   └── default_config.yaml
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后的数据
│   └── splits/            # 数据集划分
├── src/                   # 源代码
│   ├── __init__.py
│   ├── data/              # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py     # 数据集类
│   │   └── preprocessor.py # 数据预处理
│   ├── models/            # 模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py  # 基础模型类
│   │   ├── lstm_model.py  # LSTM模型
│   │   └── transformer_model.py # Transformer模型
│   ├── training/          # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py     # 训练器
│   │   └── utils.py       # 训练工具
│   └── utils/             # 通用工具
│       ├── __init__.py
│       ├── logger.py      # 日志记录
│       └── metrics.py     # 评估指标
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── inference.py           # 推理脚本
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 🆕 对比学习训练（推荐）

#### 快速演示
```bash
# 运行对比学习演示（自动生成示例数据）
python demo_contrastive.py

# 或使用快速启动脚本
./quickstart_contrastive.sh
```

#### 完整对比学习训练
```bash
# 对比学习预训练 + 有监督微调
python train_contrastive.py \
    --config configs/contrastive_config.yaml \
    --data_path data/your_dataset \
    --experiment_name toa_contrastive \
    --mode both

# 仅对比学习预训练
python train_contrastive.py \
    --config configs/contrastive_config.yaml \
    --data_path data/your_dataset \
    --experiment_name toa_pretrain \
    --mode contrastive

# 传统有监督训练（基准对比）
python train_contrastive.py \
    --config configs/contrastive_config.yaml \
    --data_path data/your_dataset \
    --experiment_name toa_supervised \
    --mode supervised
```

#### 对比实验
```bash
# 运行对比学习 vs 传统训练的性能对比
python comparison_experiment.py
```

### 传统训练方法

#### 生成示例数据
```bash
python generate_cir_toa_data.py
```

#### 训练模型
```bash
python train.py --config configs/unified_config.yaml --data_path data/raw/cir_toa_train.csv --experiment_name cir_toa_experiment
```

#### 评估模型
```bash
python evaluate.py --experiment_dir results/your_experiment --data_path data/test_dataset
```

#### 推理
```bash
python inference.py --experiment_dir results/your_experiment --input_path data/new_data.csv --output_path results/predictions.csv
```

#### 快速开始
```bash
chmod +x quickstart.sh
./quickstart.sh
```

## 设备支持

项目支持多种计算设备：

- **CPU**: 所有平台支持
- **CUDA GPU**: NVIDIA显卡支持
- **Apple Silicon (MPS)**: M1/M2/M3 Mac支持

### 设备检测和性能测试

运行设备测试脚本来检测你的设备支持情况：

```bash
python test_device.py
```

### Mac用户特别说明

如果你使用Apple Silicon Mac (M1/M2/M3)，确保安装支持MPS的PyTorch：

```bash
pip3 install torch torchvision torchaudio
```

然后在配置文件中设置：
```yaml
device: "mps"  # 或者使用 "auto" 自动检测最优设备
```

## 扩展性

- 可以轻松添加新的模型架构到`src/models/`目录
- 数据预处理可以通过修改`src/data/preprocessor.py`来自定义
- 评估指标可以在`src/utils/metrics.py`中添加
- 训练策略可以在`src/training/trainer.py`中修改
