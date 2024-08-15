import torch
import torch.nn as nn
import torchvision.models as models


class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()

        # Instantiate ResNet-34
        resnet34 = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-1])

        self.acceleration_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=resnet34.fc.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

        self.steering_angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=resnet34.fc.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.resnet34(x)
        acceleration = torch.clip(self.acceleration_head(features),-1,1)
        throttle = torch.clip(acceleration,0,1)
        brake = torch.clip(acceleration,-1,0)
        steering_angle = torch.clip(self.steering_angle_head(features),-1,1)

        return {'acceleration': throttle[:,0], 'steering_angle': steering_angle[:,0], 'brake': brake[:,0], 'acceleration_continuous': acceleration[:,0]}

if __name__ == "__main__":
    # Test the model with random input
    model = ResnetModel()
    input_tensor = torch.randn((1, 3, 256, 256))  # Batch size of 1, 1 channel, 256x256 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor)
