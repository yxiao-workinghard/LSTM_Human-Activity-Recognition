"""
Simple example script demonstrating how to use the HAR modules
"""
import torch
from data_loader import prepare_data
from model import create_model
from trainer import train_model
from evaluator import evaluate_model


def simple_example():
    """
    Simple example of using the HAR system
    """
    print("Human Activity Recognition - Simple Example")
    print("=" * 50)
    
    # Configuration
    DATA_PATH = 'UCIDataset/'  # Update this path as needed
    BATCH_SIZE = 64
    EPOCHS = 10  # Small number for quick testing
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        train_loader, test_loader, data_info = prepare_data(
            data_path=DATA_PATH, 
            batch_size=BATCH_SIZE
        )
        print("✓ Data loaded successfully!")
        
        # 2. Create model
        print("\n2. Creating model...")
        model = create_model(
            input_size=data_info['n_features'],
            hidden_size=64,  # Smaller for quick testing
            num_classes=data_info['n_outputs'],
            device=device
        )
        print("✓ Model created!")
        
        # 3. Train model
        print(f"\n3. Training model for {EPOCHS} epochs...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            epochs=EPOCHS,
            learning_rate=0.001,
            verbose=True,
            save_path='example_model.pth'
        )
        print("✓ Training completed!")
        
        # 4. Evaluate model
        print("\n4. Evaluating model...")
        results = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            show_plots=False  # Set to True if you want to see plots
        )
        print("✓ Evaluation completed!")
        
        print(f"\nFinal Results:")
        print(f"- Accuracy: {results['accuracy']:.2f}%")
        print(f"- Model saved as: example_model.pth")
        
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        print("Please update the DATA_PATH variable or ensure the UCI HAR Dataset is available.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    simple_example()