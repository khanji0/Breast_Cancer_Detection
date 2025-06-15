import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class BreastCancerClassifier(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=2):
        super(BreastCancerClassifier, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            in_features = self.model.fc.in_features
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            in_features = self.model.classifier.in_features
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace the final layer
        if model_name == 'resnet50':
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'densenet121':
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == 'efficientnet_b0':
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Get class predictions"""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs

def get_model(model_name='resnet50', pretrained=True, num_classes=2, device='cuda'):
    """
    Create and initialize a model
    
    Args:
        model_name (str): Name of the model architecture
        pretrained (bool): Whether to use pretrained weights
        num_classes (int): Number of output classes
        device (str): Device to place the model on
    
    Returns:
        BreastCancerClassifier: Initialized model
    """
    model = BreastCancerClassifier(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # Move model to device
    model = model.to(device)
    
    return model

def load_model(model_path, model_name='resnet50', num_classes=2, device='cuda'):
    """
    Load a trained model from disk
    
    Args:
        model_path (str): Path to the saved model
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        device (str): Device to place the model on
    
    Returns:
        BreastCancerClassifier: Loaded model
    """
    model = get_model(model_name, pretrained=False, num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different model architectures
    for model_name in ['resnet50', 'densenet121', 'efficientnet_b0']:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, device=device)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test prediction
        preds, probs = model.predict(x)
        print(f"Predictions shape: {preds.shape}")
        print(f"Probabilities shape: {probs.shape}") 