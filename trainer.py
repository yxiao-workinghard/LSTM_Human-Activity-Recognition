"""
Training module for LSTM Human Activity Recognition model
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ModelTrainer:
    """
    Trainer class for the LSTM model
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model to train
            device (str): Device to use for training ('cpu' or 'cuda')
            learning_rate (float): Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def train_epoch(self, train_loader):
        """
        Train the model for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """
        Validate the model for one epoch
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, verbose=True, save_path='lstm_weights.pth'):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs (int): Number of epochs to train
            verbose (bool): Whether to print training progress
            save_path (str): Path to save the best model
            
        Returns:
            dict: Training history
        """
        best_val_accuracy = 0.0
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), save_path)
                if verbose:
                    print(f'New best model saved with validation accuracy: {val_accuracy:.2f}%')
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        print(f'\nTraining completed! Best validation accuracy: {best_val_accuracy:.2f}%')
        return self.history
    
    def get_history(self):
        """
        Get the training history
        
        Returns:
            dict: Training history
        """
        return self.history


def train_model(model, train_loader, val_loader, device, epochs=50, learning_rate=0.001, 
                verbose=True, save_path='lstm_weights.pth'):
    """
    Convenience function to train a model
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device (str): Device to use for training
        epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
        verbose (bool): Whether to print training progress
        save_path (str): Path to save the best model
        
    Returns:
        dict: Training history
    """
    trainer = ModelTrainer(model, device, learning_rate)
    history = trainer.train(train_loader, val_loader, epochs, verbose, save_path)
    return history