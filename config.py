"""
Configuration file for Human Activity Recognition LSTM model
"""

# Data configuration
DATA_CONFIG = {
    'data_path': 'UCIDataset/',
    'batch_size': 128,
    'num_classes': 6,
    'class_names': ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
}

# Model configuration
MODEL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': 0.5,
}

# Training configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'save_path': 'lstm_weights.pth',
    'verbose': True
}

# Evaluation configuration
EVAL_CONFIG = {
    'show_plots': True,
    'save_results': False,
}

# Device configuration
DEVICE_CONFIG = {
    'use_cuda': True,  # Use GPU if available
    'cuda_device': 0   # CUDA device index
}