# Breast Cancer Detection using Deep Learning

This project implements a deep learning solution for detecting Invasive Ductal Carcinoma (IDC) in breast histopathology images. The model is trained on a dataset of 277,524 image patches, with 198,738 IDC negative and 78,786 IDC positive samples.

## Project Structure

```
breast_cancer_detection/
├── data/
│   ├── archive/          # Original dataset
│   ├── processed/        # Preprocessed data
│   └── models/          # Saved models
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Data preprocessing
│   │   └── dataloader.py     # Data loading utilities
│   ├── models/
│   │   ├── model.py         # Model architecture
│   │   └── train.py         # Training script
│   ├── utils/
│   │   └── visualization.py  # Visualization utilities
│   ├── config.py            # Configuration settings
│   └── main.py             # Main pipeline script
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA notebook
│   └── model_evaluation.ipynb          # Evaluation notebook
├── tests/
│   └── test_model.py       # Model tests
├── evaluation_results/     # Evaluation outputs
├── requirements.txt        # Project dependencies
├── setup.py               # Installation script
└── README.md             # Project documentation
```

## Features

- **Data Preprocessing**:
  - Image normalization and standardization
  - Data augmentation for training
  - Train/validation/test split
  - Class imbalance handling

- **Model Architecture**:
  - Support for multiple architectures (ResNet50, DenseNet121, EfficientNet-B0)
  - Transfer learning with pretrained weights
  - Custom classification head
  - Dropout for regularization

- **Training Pipeline**:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Training history tracking
  - GPU support

- **Evaluation and Visualization**:
  - Comprehensive metrics (accuracy, precision, recall, F1-score, ROC-AUC)
  - Confusion matrix
  - ROC and PR curves
  - Training history plots
  - Error analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### 1. Data Preprocessing

```bash
python src/main.py --mode preprocess
```

This will:
- Process the raw images
- Apply data augmentation
- Split the data into train/validation/test sets
- Save the processed data

### 2. Model Training

```bash
python src/main.py --mode train --model resnet50 --batch-size 32 --epochs 50
```

Options:
- `--model`: Model architecture (resnet50, densenet121, efficientnet_b0)
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--weight-decay`: Weight decay for regularization
- `--patience`: Patience for early stopping

### 3. Model Evaluation

```bash
python src/main.py --mode evaluate
```

This will:
- Load the best model
- Evaluate on the test set
- Generate performance metrics
- Create visualizations
- Save results

### 4. Jupyter Notebooks

- `notebooks/exploratory_data_analysis.ipynb`: Data exploration and analysis
- `notebooks/model_evaluation.ipynb`: Model evaluation and visualization

## Model Architecture

The project supports multiple model architectures:

1. **ResNet50**:
   - Pretrained on ImageNet
   - Custom classification head
   - Dropout for regularization

2. **DenseNet121**:
   - Pretrained on ImageNet
   - Custom classification head
   - Dropout for regularization

3. **EfficientNet-B0**:
   - Pretrained on ImageNet
   - Custom classification head
   - Dropout for regularization

## Evaluation Metrics

The model is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

## Visualization Components

The project includes various visualizations:

- Confusion matrix
- ROC curve
- Precision-Recall curve
- Training history plots
- Class distribution
- Sample predictions
- Error analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Breast Histopathology Images](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)
- PyTorch team for the deep learning framework
- All contributors and users of this project 