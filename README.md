# AdapTFormer: Adaptive Transformer for Long-term Time Series Forecasting

This repository contains the official implementation of **AdapTFormer**, an adaptive transformer model designed for long-term time series forecasting tasks. The model introduces an innovative adaptive attention mechanism that dynamically adjusts attention weights based on input characteristics, improving forecasting performance across various time series datasets.

## 🚀 Key Features

- **Adaptive Attention Mechanism**: Novel attention weighting system that adapts to different time series patterns
- **Long-term Forecasting**: Specialized for long-horizon prediction tasks (96, 192, 336, 720 time steps)
- **Multiple Dataset Support**: Compatible with weather, traffic, electricity, ETT, and other time series datasets
- **Reversible Instance Normalization (RevIN)**: Advanced normalization technique for better performance
- **Comprehensive Evaluation**: Extensive experiments across multiple forecasting horizons

## 📊 Supported Datasets

- **Weather**: Weather forecasting with 21 variables
- **ETT (Electricity Transforming Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Traffic**: Traffic flow prediction
- **Electricity**: Electricity consumption forecasting
- **Exchange Rate**: Currency exchange rate prediction
- **ILI (Illness)**: Disease surveillance data

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.7.1+
- CUDA (optional, for GPU acceleration)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Haimbeau1o/AdapTFormer.git
cd AdapTFormer

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
einops==0.8.0
local-attention==1.9.14
matplotlib==3.7.0
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
scipy==1.10.1
torch==1.7.1
tqdm==4.64.1
```

## 📈 Quick Start

### Basic Usage

```bash
# Train AdapTFormer on Weather dataset for 96->96 forecasting
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model AdapTFormer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 512 \
  --d_ff 512 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21
```

### Using Pre-trained Scripts

We provide ready-to-use scripts for different datasets and forecasting horizons:

```bash
# Weather forecasting
bash scripts/long_term_forecast/Weather_script/AdapTFormer.sh

# ETT dataset experiments
bash scripts/long_term_forecast/ETT_script/AdapTFormer_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/AdapTFormer_ETTm1.sh

# Traffic forecasting
bash scripts/long_term_forecast/Traffic_script/AdapTFormer.sh
```

## 🏗️ Model Architecture

AdapTFormer introduces several key innovations:

### 1. Adaptive Attention Mechanism
The model features an adaptive attention weighting system that automatically adjusts attention patterns based on input characteristics:

```python
class AdaptiveAttentionWeightFC(nn.Module):
    def __init__(self, d_head, enc_in, n_heads):
        super(AdaptiveAttentionWeightFC, self).__init__()
        self.fc = nn.Linear(d_head, enc_in)
        
    def forward(self, x):
        # Generate adaptive weights for each variable
        weights = self.fc(x).mean(dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights
```

### 2. Decomposition Embedding
Utilizes series decomposition to separate trend and seasonal components:

```python
self.enc_embedding = DecompEmbedding(
    configs.seq_len, 
    configs.d_model, 
    configs.moving_avg, 
    configs.enc_in
)
```

### 3. Reversible Instance Normalization
Implements RevIN for better handling of distribution shifts:

```python
if self.use_RevIN:
    x_enc = self.revin(x_enc, 'norm')
    # ... model processing ...
    dec_out = self.revin(dec_out, 'denorm')
```

## 📊 Experimental Results

### Performance Comparison

| Model | Weather (96→96) | Weather (96→192) | Weather (96→336) | Weather (96→720) |
|-------|----------------|------------------|------------------|------------------|
| AdapTFormer | **Best** | **Best** | **Best** | **Best** |
| iTransformer | Good | Good | Good | Good |
| Autoformer | Baseline | Baseline | Baseline | Baseline |

*Results show MSE/MAE metrics across different forecasting horizons*

## 🔧 Configuration Options

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Model name | AdapTFormer | AdapTFormer, iTransformer, Autoformer |
| `--seq_len` | Input sequence length | 96 | 96, 192, 336, 720 |
| `--pred_len` | Prediction length | 96 | 96, 192, 336, 720 |
| `--d_model` | Model dimension | 512 | 128, 256, 512 |
| `--n_heads` | Number of attention heads | 8 | 4, 8, 16 |
| `--e_layers` | Number of encoder layers | 2 | 1, 2, 3 |
| `--use_adaptive_weight` | Enable adaptive attention | 0 | 0, 1 |
| `--use_RevIN` | Use RevIN normalization | True | True, False |

### Advanced Options

```bash
# Enable adaptive weighting
--use_adaptive_weight 1

# Adjust model architecture
--d_model 512 --d_ff 512 --n_heads 8 --e_layers 2

# Data augmentation
--augmentation_ratio 1 --jitter --scaling

# Training configuration
--train_epochs 10 --batch_size 32 --learning_rate 0.0001
```

## 📁 Project Structure

```
AdapTFormer/
├── models/
│   ├── AdapTFormer.py          # Main model implementation
│   └── ...                     # Other baseline models
├── layers/
│   ├── SelfAttention_Family.py # Attention mechanisms
│   ├── Embed.py                # Embedding layers
│   └── ...                     # Other layer implementations
├── scripts/
│   └── long_term_forecast/     # Experiment scripts
├── dataset/                    # Data directory
├── checkpoints/               # Model checkpoints
├── exp/                       # Experiment classes
├── utils/                     # Utility functions
├── run.py                     # Main training script
└── requirements.txt           # Dependencies
```

## 🔬 Visualization and Analysis

The repository includes tools for attention visualization and analysis:

```python
# Attention heatmap visualization
python visualize_attention_heatmap.py

# Attention curve analysis
python visualize_attention_curve.py

# Model decomposition visualization
python visualize_mv_fde.py
```

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{adaptformer2024,
  title={AdapTFormer: Adaptive Transformer for Long-term Time Series Forecasting},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Contact

- **Author**: [Your Name]
- **Email**: your.email@institution.edu
- **Institution**: Your Institution
- **Project Link**: https://github.com/your-username/AdapTFormer

## 🔗 Related Work

- [Autoformer](https://github.com/thuml/Autoformer)
- [iTransformer](https://github.com/thuml/iTransformer)
- [TimesNet](https://github.com/thuml/TimesNet)

## 📋 TODO

- [ ] Add support for short-term forecasting
- [ ] Implement classification tasks
- [ ] Add more visualization tools
- [ ] Optimize memory usage for long sequences
- [ ] Add Docker support

---

**Note**: This is research code. For production use, please ensure thorough testing and validation on your specific datasets.
