import os
import logging
import logging.config
import argparse
from datetime import datetime

from src.config import *
from src.data.preprocessing import DataPreprocessor
from src.data.dataloader import get_data_loaders, get_class_weights
from src.models.model import get_model
from src.models.train import train
from src.utils.visualization import evaluate_model, visualize_sample_predictions

def setup_logging():
    """Setup logging configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Breast Cancer Detection Pipeline')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['preprocess', 'train', 'evaluate'],
                      help='Pipeline mode: preprocess, train, or evaluate')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                      choices=['resnet50', 'densenet121', 'efficientnet_b0'],
                      help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                      help='Weight decay')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                      help='Patience for early stopping')
    return parser.parse_args()

def preprocess_data(logger):
    """Preprocess the data"""
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(DATA_DIR, PROCESSED_DATA_DIR)
    
    # Organize data
    preprocessor.organize_data(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # Get and log class distribution
    distribution = preprocessor.get_class_distribution()
    logger.info("Class distribution after preprocessing:")
    for split, counts in distribution.items():
        logger.info(f"{split.capitalize()}:")
        logger.info(f"  Class 0 (IDC negative): {counts['0']}")
        logger.info(f"  Class 1 (IDC positive): {counts['1']}")
    
    logger.info("Data preprocessing completed!")

def train_model(args, logger):
    """Train the model"""
    logger.info("Starting model training...")
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODEL_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    model, history = train(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_dir=save_dir
    )
    
    logger.info(f"Model training completed! Results saved in {save_dir}")
    return model, save_dir

def evaluate_model_pipeline(args, logger):
    """Evaluate the model"""
    logger.info("Starting model evaluation...")
    
    # Get the latest model directory
    model_dirs = sorted([d for d in os.listdir(MODEL_DIR) 
                        if os.path.isdir(os.path.join(MODEL_DIR, d))])
    if not model_dirs:
        raise ValueError("No trained models found!")
    
    latest_model_dir = os.path.join(MODEL_DIR, model_dirs[-1])
    model_path = os.path.join(latest_model_dir, 'best_model.pth')
    
    # Create evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(EVALUATION_DIR, timestamp)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get data loaders
    _, _, test_loader = get_data_loaders(
        PROCESSED_DATA_DIR,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )
    
    # Load model
    model = get_model(args.model, pretrained=False, device=DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, DEVICE, eval_dir)
    
    # Visualize sample predictions
    visualize_sample_predictions(model, test_loader, DEVICE, eval_dir)
    
    logger.info(f"Model evaluation completed! Results saved in {eval_dir}")
    return metrics

def main():
    """Main function to run the pipeline"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Breast Cancer Detection Pipeline")
    
    # Parse arguments
    args = parse_args()
    
    try:
        if args.mode == 'preprocess':
            preprocess_data(logger)
        elif args.mode == 'train':
            model, save_dir = train_model(args, logger)
        elif args.mode == 'evaluate':
            metrics = evaluate_model_pipeline(args, logger)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 