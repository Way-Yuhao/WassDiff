import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), resnet_style=False):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, activation)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.resnet_style = resnet_style

    def forward(self, x):
        identity = x
        x = self.conv_block(x)
        if self.resnet_style:
            x = torch.cat([x, identity], dim=1)
        x = self.pool(x)
        return x, identity


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, concat_channels, out_channels, activation=nn.ReLU()):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels + concat_channels, out_channels, activation)

    def forward(self, x, concat_tensor):
        x = self.upconv(x)
        x = torch.cat([x, concat_tensor], dim=1)
        x = self.conv_block(x)
        return x


class SEVIRUNet(nn.Module):
    def __init__(self, start_filters=32):
        super(SEVIRUNet, self).__init__()
        self.start_filters = start_filters
        # Encoder Blocks
        self.encoder0 = EncoderBlock(8, start_filters)
        self.encoder1 = EncoderBlock(start_filters, start_filters * 2)
        self.encoder2 = EncoderBlock(start_filters * 2, start_filters * 4)
        self.encoder3 = EncoderBlock(start_filters * 4, start_filters * 8)
        # Center Convolution Block
        self.center = ConvBlock(start_filters * 8, start_filters * 16)
        # Decoder Blocks
        # self.decoder3 = DecoderBlock(start_filters * 16, start_filters * 8, start_filters * 8)
        # self.decoder2 = DecoderBlock(start_filters * 8, start_filters * 4, start_filters * 4)
        # self.decoder1 = DecoderBlock(start_filters * 4, start_filters * 2, start_filters * 2)
        # self.decoder0 = DecoderBlock(start_filters * 2, start_filters, start_filters)
        # # Final Convolution to map to output channels
        # self.final_conv = nn.Conv2d(start_filters, 1, kernel_size=1)

        # Decoder Blocks
        self.decoder3 = DecoderBlock(start_filters * 16, start_filters * 4, start_filters * 8)
        self.decoder2 = DecoderBlock(start_filters * 8, start_filters * 2, start_filters * 4)
        self.decoder1 = DecoderBlock(start_filters * 4, start_filters, start_filters * 2)
        self.decoder0 = DecoderBlock(start_filters * 2, 8, start_filters)
        # Final Convolution to map to output channels
        self.final_conv = nn.Conv2d(start_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        encoder0_pool, encoder0 = self.encoder0(x)
        encoder1_pool, encoder1 = self.encoder1(encoder0_pool)
        encoder2_pool, encoder2 = self.encoder2(encoder1_pool)
        encoder3_pool, encoder3 = self.encoder3(encoder2_pool)

        # Center block
        center = self.center(encoder3_pool)

        # Decoder Path
        decoder3 = self.decoder3(center, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        decoder0 = self.decoder0(decoder1, encoder0)

        # Final output layer
        output = self.final_conv(decoder0)
        return output

