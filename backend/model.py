import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Map latent vector to feature map size
        self.fc = nn.Linear(2, 64 * 7 * 7)
        
        # Use Upsample (Linear/Bilinear interpolation) to scale up instead of Deconv
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 7, 7)
        # Upsample -> Conv structure
        x = F.relu(self.conv1(self.upsample1(x)))
        x = torch.tanh(self.conv2(self.upsample2(x))) # Output range -1 to 1
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # We only need decoder for inference in this app, 
        # but including full class if needed for loading state_dict strictly
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(x)
