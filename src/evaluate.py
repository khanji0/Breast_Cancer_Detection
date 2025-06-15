import os
import torch
import json
from models.model import get_model
from data.data_loader import create_data_loaders
from utils.visualization import (
    evaluate_model,
    visualize_sample_predictions,
    plot_training_history,
    save_evaluation_results
)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    data_dir = "data/archive"
    _, _, test_loader = create_data_loaders(
        data_dir, batch_size=32, val_split=0.2, test_split=0.1
    )
    
    # Load model
    model = get_model(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Load training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    # Evaluate model and generate visualizations
    print("Evaluating model...")
    report_df = evaluate_model(model, test_loader, device)
    
    # Visualize sample predictions
    print("Generating sample predictions visualization...")
    visualize_sample_predictions(model, test_loader, device)
    
    # Plot training history
    print("Generating training history plots...")
    plot_training_history(history)
    
    # Save all results
    print("Saving evaluation results...")
    save_evaluation_results(report_df, history)
    
    # Print classification report
    print("\nClassification Report:")
    print(report_df)
    
    print("\nEvaluation complete! All results saved to 'evaluation_results/' directory.")

if __name__ == "__main__":
    main() 