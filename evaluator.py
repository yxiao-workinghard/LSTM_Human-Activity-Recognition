"""
Evaluation and visualization module for LSTM Human Activity Recognition model
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    """
    Evaluator class for the LSTM model
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the evaluator
        
        Args:
            model: PyTorch model to evaluate
            device (str): Device to use for evaluation ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.class_names = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                           'SITTING', 'STANDING', 'LAYING']
    
    def predict(self, data_loader):
        """
        Make predictions on a dataset
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            tuple: (predictions, true_labels)
        """
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def evaluate(self, data_loader, model_path=None):
        """
        Evaluate the model on a dataset
        
        Args:
            data_loader: DataLoader for the dataset
            model_path (str): Path to saved model weights (optional)
            
        Returns:
            dict: Evaluation results
        """
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        
        predictions, true_labels = self.predict(data_loader)
        
        # Calculate accuracy
        accuracy = 100 * np.sum(predictions == true_labels) / len(true_labels)
        
        # Generate classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=self.class_names, 
                                     digits=4, 
                                     output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return results
    
    def print_evaluation_report(self, results):
        """
        Print evaluation results in a formatted way
        """
        print(f"Best Validation Accuracy: {results['accuracy']:.2f}%")
        print("\nClassification Report:")
        
        # Convert dict back to string format for printing
        report_dict = results['classification_report']
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for class_name in self.class_names:
            if class_name.lower().replace('_', ' ') in report_dict:
                class_key = class_name.lower().replace('_', ' ')
            else:
                # Find the matching key in report_dict
                class_key = None
                for key in report_dict.keys():
                    if isinstance(report_dict[key], dict) and key != 'accuracy' and key != 'macro avg' and key != 'weighted avg':
                        class_key = key
                        break
            
            if class_key and class_key in report_dict:
                metrics = report_dict[class_key]
                print(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                      f"{metrics['f1-score']:<10.4f} {metrics['support']:<10}")
        
        # Print overall metrics
        if 'macro avg' in report_dict:
            avg = report_dict['macro avg']
            print(f"{'Macro avg':<20} {avg['precision']:<10.4f} {avg['recall']:<10.4f} "
                  f"{avg['f1-score']:<10.4f} {avg['support']:<10}")
        
        if 'weighted avg' in report_dict:
            avg = report_dict['weighted avg']
            print(f"{'Weighted avg':<20} {avg['precision']:<10.4f} {avg['recall']:<10.4f} "
                  f"{avg['f1-score']:<10.4f} {avg['support']:<10}")
    


def evaluate_model(model, data_loader, device='cpu', model_path=None):
    """
    Convenience function to evaluate a model
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device (str): Device to use for evaluation
        model_path (str): Path to saved model weights (optional)
        
    Returns:
        dict: Evaluation results
    """
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(data_loader, model_path)
    
    evaluator.print_evaluation_report(results)
    
    return results