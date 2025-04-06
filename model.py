import torch
import torch.nn as nn
import torch.nn.functional as F


class SolarRadiationPredictor(nn.Module):
    def __init__(self):
        super(SolarRadiationPredictor, self).__init__()
        # Image branch: Convolutional Neural Network

        # Block 1:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # (3,540,960) -> (32,540,960)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples by 2: (32,270,480)

        # Block 2:
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64,270,480)
        # After pooling: (64,135,240)

        # Block 3:
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (128,135,240)
        # After pooling: (128,67,120)  [since 135/2 ≈ 67 (floor) and 240/2 = 120]

        # Block 4:
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # (256,67,120)
        # After pooling: (256,33,60) [67/2 ≈ 33, 120/2 = 60]

        # Fully-connected layer for the image branch.
        # The flattened feature size is: 256 * 33 * 60.
        self.fc_img = nn.Linear(256 * 33 * 60, 128)

        # Sensor branch: Fully connected layer for the solar radiation sensor (a single value)
        self.fc_sensor = nn.Linear(1, 16)

        # Fusion branch: Combine image and sensor features and predict a scalar output.
        # The concatenated feature size is 128 + 16 = 144.
        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)  # Final output for regression (predicting solar radiation)

    def forward(self, image, sensor):
        # Image branch forward pass:
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = F.relu(self.fc_img(x))  # Resulting shape: (batch_size, 128)

        # Sensor branch forward pass:
        # Ensure sensor data is of shape (batch_size, 1)
        sensor = sensor.unsqueeze(1) if sensor.dim() == 1 else sensor
        s = F.relu(self.fc_sensor(sensor))  # Resulting shape: (batch_size, 16)

        # Concatenate both branches:
        combined = torch.cat((x, s), dim=1)  # Shape: (batch_size, 128+16=144)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        output = self.out(combined)  # Final shape: (batch_size, 1)
        return output


# Quick test to check model dimensions:
if __name__ == "__main__":
    model = SolarRadiationPredictor()
    dummy_image = torch.randn(8, 3, 540, 960)  # Simulate a batch of 8 images with shape (3,540,960)
    dummy_sensor = torch.randn(8)  # Simulate a batch of 8 sensor readings
    output = model(dummy_image, dummy_sensor)
    print("Output shape:", output.shape)  # Expected: torch.Size([8, 1])