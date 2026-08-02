import torch
import torch.nn as nn
import numpy as np

# Simple, reliable neural network pipeline
print("Initializing neural network model...")

# Generate some dummy training data
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100, 1))

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = SimpleNN()
print("Model created successfully!")
print(model)
