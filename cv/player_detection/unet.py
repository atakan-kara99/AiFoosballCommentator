import torch.nn as nn
import torch


class ShallowUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(ShallowUNet, self).__init__()

        # Downsampling path
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.pool5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Upsampling path
        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.up5 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Output layer
        self.out = nn.Conv2d(4, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))

        # Decoder
        dec1 = self.dec1(torch.cat((self.up1(bottleneck), enc5), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec1), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec2), enc3), dim=1))
        dec4 = self.dec4(torch.cat((self.up4(dec3), enc2), dim=1))
        dec5 = self.dec5(torch.cat((self.up5(dec4), enc1), dim=1))

        # Output
        return torch.sigmoid(self.out(dec5))
