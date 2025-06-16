# File: tests/test_model.py
# Description: Unit tests for breast cancer detection model

import os
import sys
import unittest
import torch
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import get_model, BreastCancerClassifier
from src.config import DEVICE

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.input_channels = 3
        self.image_size = 224
        self.num_classes = 2
        
        # Create dummy input
        self.dummy_input = torch.randn(
            self.batch_size,
            self.input_channels,
            self.image_size,
            self.image_size
        ).to(DEVICE)
    
    def test_model_creation(self):
        """Test model creation with different architectures"""
        model_names = ['resnet50', 'densenet121', 'efficientnet_b0']
        
        for model_name in model_names:
            with self.subTest(model_name=model_name):
                model = get_model(model_name, device=DEVICE)
                self.assertIsInstance(model, BreastCancerClassifier)
    
    def test_forward_pass(self):
        """Test forward pass through the model"""
        model = get_model('resnet50', device=DEVICE)
        output = model(self.dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_prediction(self):
        """Test model prediction method"""
        model = get_model('resnet50', device=DEVICE)
        preds, probs = model.predict(self.dummy_input)
        
        # Check predictions shape
        self.assertEqual(preds.shape, (self.batch_size,))
        
        # Check probabilities shape
        self.assertEqual(probs.shape, (self.batch_size, self.num_classes))
        
        # Check probability values are between 0 and 1
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        
        # Check probabilities sum to 1
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(self.batch_size).to(DEVICE)))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create and save model
        model = get_model('resnet50', device=DEVICE)
        save_path = 'test_model.pth'
        torch.save(model.state_dict(), save_path)
        
        # Load model
        loaded_model = get_model('resnet50', pretrained=False, device=DEVICE)
        loaded_model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        
        # Compare outputs
        with torch.no_grad():
            original_output = model(self.dummy_input)
            loaded_output = loaded_model(self.dummy_input)
        
        self.assertTrue(torch.allclose(original_output, loaded_output))
        
        # Clean up
        os.remove(save_path)

if __name__ == '__main__':
    unittest.main() 