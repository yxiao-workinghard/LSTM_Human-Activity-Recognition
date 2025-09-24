# Human Activity Recognition with PyTorch LSTM

This project implements a Human Activity Recognition (HAR) system using LSTM neural networks with PyTorch. The model classifies human activities based on accelerometer and gyroscope sensor data from smartphones.

## Project Structure

```
├── main.py                 # Main script to run training and evaluation
├── data_loader.py          # Data loading and preprocessing functions
├── model.py               # LSTM model definition
├── trainer.py             # Training logic and utilities
├── evaluator.py           # Evaluation and visualization functions
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
└── README_LSTM.md         # This file
```

## Features

- **Modular Design**: Clean separation of data loading, model definition, training, and evaluation
- **PyTorch Implementation**: Modern deep learning framework with GPU support
- **Comprehensive Evaluation**: Detailed metrics, confusion matrix, and training visualizations
- **Command-line Interface**: Easy-to-use CLI with configurable parameters
- **Model Persistence**: Save and load trained models

## Dataset

The project uses the UCI Human Activity Recognition Dataset, which contains sensor data from 30 volunteers performing six activities:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

## Requirements

```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation

1. Clone or download this project
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the UCI HAR Dataset is available in the `UCIDataset/` folder

## Usage

### Basic Usage

Run with default parameters:
```bash
python main.py
```

### Custom Configuration

Train only:
```bash
python main.py --mode train --epochs 100 --learning_rate 0.001
```

Evaluate only (requires pre-trained model):
```bash
python main.py --mode eval --model_path lstm_weights.pth
```

With custom model architecture:
```bash
python main.py --hidden_size 256 --num_layers 2 --dropout 0.3
```

With visualization:
```bash
python main.py --show_plots
```

### Command Line Arguments

- `--data_path`: Path to UCI HAR Dataset (default: 'UCIDataset/')
- `--batch_size`: Batch size for training (default: 128)
- `--hidden_size`: LSTM hidden size (default: 128)
- `--num_layers`: Number of LSTM layers (default: 1)
- `--dropout`: Dropout probability (default: 0.5)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--mode`: Mode - 'train', 'eval', or 'both' (default: 'both')
- `--model_path`: Path to save/load model (default: 'lstm_weights.pth')
- `--show_plots`: Show training history and confusion matrix
- `--save_results`: Save evaluation results to file

## Model Architecture

The LSTM model consists of:

1. **LSTM Layer**: Processes sequential sensor data
2. **Dropout Layers**: Regularization to prevent overfitting
3. **Fully Connected Layers**: 
   - Hidden layer with 64 units and ReLU activation
   - Output layer with 6 units (one per activity class)

## API Usage

You can also use the modules programmatically:

```python
from data_loader import prepare_data
from model import create_model
from trainer import train_model
from evaluator import evaluate_model

# Load data
train_loader, test_loader, data_info = prepare_data('UCIDataset/')

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(data_info['n_features'], device=device)

# Train model
history = train_model(model, train_loader, test_loader, device)

# Evaluate model
results = evaluate_model(model, test_loader, device)
```

## Results

The model typically achieves:
- **Accuracy**: ~95% on test set
- **Best performing activities**: Static activities (SITTING, STANDING, LAYING)
- **Challenging activities**: Walking variations (stairs vs. flat)

## Key Improvements from Original

- **Modern PyTorch**: Replaced TensorFlow/Keras with PyTorch
- **Modular Design**: Separated concerns into different modules
- **Better Error Handling**: Comprehensive exception handling
- **CLI Interface**: Easy command-line usage
- **Enhanced Visualization**: Better plots and metrics
- **GPU Support**: Automatic GPU detection and usage
- **Code Documentation**: Comprehensive docstrings and comments

## Customization

### Adding New Models

Extend the `model.py` file:

```python
class CustomLSTM(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
        pass
```

### Custom Training Logic

Extend the `trainer.py` file:

```python
class CustomTrainer(ModelTrainer):
    def custom_training_step(self, ...):
        # Your custom training logic
        pass
```

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or model size
2. **Data not found**: Ensure UCI HAR Dataset is in the correct path
3. **Import errors**: Install all required packages

## License

This project is for educational and research purposes. Please cite the UCI HAR Dataset if used in publications.