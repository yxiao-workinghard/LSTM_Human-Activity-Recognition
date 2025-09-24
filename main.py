import torch
import argparse
import sys

from data_loader import prepare_data
from model import create_model
from trainer import train_model
from evaluator import evaluate_model


def get_device():
    """Check if GPU is available and return appropriate device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory Usage:')
        print(f'Allocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
        print(f'Reserved: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')
    return device


def main(args):
    """
    Main function to run the Human Activity Recognition pipeline
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("Human Activity Recognition using LSTM")
    print("=" * 60)
    
    # Set device
    device = get_device()
    
    # Prepare data
    print("\n1. Loading and preparing data...")

    train_loader, test_loader, data_info = prepare_data(
        data_path=args.data_path, 
        batch_size=args.batch_size
    )
    print(f"✓ Data loaded successfully!")
    print(f"  - Training samples: {data_info['n_train_samples']}")
    print(f"  - Test samples: {data_info['n_test_samples']}")
    print(f"  - Features: {data_info['n_features']}")
    print(f"  - Timesteps: {data_info['n_timesteps']}")
    print(f"  - Classes: {data_info['n_outputs']}")
        
    # Create model
    print("\n2. Creating model...")
    model = create_model(
        input_size = data_info['n_features'],
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        num_classes = data_info['n_outputs'],
        dropout = args.dropout,
        device = device
    )
    print("Model created successfully!")
    
    # Training
    if args.mode in ['train', 'both']:
        print("\n3. Training model...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            verbose=True,
            save_path=args.model_path
        )
        print("Training completed!")
   
    # Evaluation
    if args.mode in ['eval', 'both']:
        print("\n4. Evaluating model...")
        results = evaluate_model(
            model = model,
            data_loader = test_loader,
            device = device,
            model_path = args.model_path if args.mode == 'eval' else None,
        )
        print("Evaluation completed!")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Human Activity Recognition using LSTM')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='D:/Public Dataset/UCI HAR Dataset/',
                        help='Path to UCI HAR Dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and evaluation (default: 128)')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size of LSTM (default: 128)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (default: 0.5)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs (default: 150)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], 
                        default='both', help='Mode: train, eval, or both (default: both)')
    parser.add_argument('--model_path', type=str, default='lstm_weights.pth',
                        help='Path to save/load model weights (default: lstm_weights.pth)')

    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Run main pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)