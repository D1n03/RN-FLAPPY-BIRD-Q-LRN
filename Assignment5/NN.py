import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        # convolutional layers
        self.conv_layers = torch.nn.Sequential(
            # input: 1 channel (grayscale image), output: 32 feature maps
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # max pooling to reduce spatial dimensions
            # input: 32 channels, output: 64 feature maps
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # input: 64 channels, output: 128 feature maps
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # input: 128 channels, output: 256 feature maps
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Flatten(), # flatten the output of convolutional layers
            # fully connected layer: Input size is 256 * 5 * 5, Output size is 512
            torch.nn.Linear(256 * 5 * 5, 512),
            torch.nn.ReLU(),
            # output layer: Input size is 512, Output size is defined by output_size
            torch.nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.conv_layers(x) # pass input through convolutional layers
        x = x.view(x.size(0), -1) # flatten before fully connected layers
        x = self.linear_layers(x) # pass through fully connected layers
        return x