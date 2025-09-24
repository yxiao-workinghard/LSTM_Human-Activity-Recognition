import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    LSTM-based neural network for human activity recognition
    
    Architecture:
    - LSTM layer with configurable hidden size and number of layers
    - Dropout for regularization
    - Two fully connected layers with ReLU activation
    - Output layer for classification
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=6, dropout=0.5):
        """
        Initialize the LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output from the sequence
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout1(out)
        
        # Pass through first fully connected layer
        out = F.relu(self.fc1(out))
        
        # Apply dropout
        out = self.dropout2(out)
        
        # Pass through second fully connected layer (output layer)
        out = self.fc2(out)
        
        return out
    
    def get_model_info(self):
        """
        Get information about the model architecture
        
        Returns:
            dict: Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }


def create_model(input_size, hidden_size=128, num_layers=1, num_classes=6, dropout=0.5, device='cpu'):
    """
    Create and initialize the LSTM model
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of features in the hidden state
        num_layers (int): Number of recurrent layers
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
        device (str): Device to move the model to ('cpu' or 'cuda')
        
    Returns:
        LSTMModel: Initialized model
    """
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout)
    model = model.to(device)
    
    print("Model Architecture:")
    print(model)
    
    model_info = model.get_model_info()
    print(f"\nTotal parameters: {model_info['total_parameters']}")
    print(f"Trainable parameters: {model_info['trainable_parameters']}")
    
    return model