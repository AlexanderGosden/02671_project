import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, H: int, W: int, latent_dim: int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.latent_dim = latent_dim
        self.encoder_net = self.__get_encoder_net()

    def __get_encoder_net(self):
        encoder_net = nn.Sequential(
            nn.Flatten(),  # flatten image [N x H*W]
            nn.Linear(self.H * self.W, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.latent_dim) 
        )
        return encoder_net

    def forward(self, x: torch.tensor):
        return self.encoder_net(x)


class Decoder(nn.Module):
    def __init__(self, H: int, W: int, latent_dim: int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.latent_dim = latent_dim
        self.decoder_net = self.__get_decoder_net()
        self.sigmoid = nn.Sigmoid()

    def __get_decoder_net(self):
        decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.H * self.W), 
            nn.Sigmoid()  # Sigmoid to map the output to [0, 1] range
        )
        return decoder_net

    def forward(self, z: torch.tensor):
        logits = self.decoder_net(z)  # [N x H*W]
        logits = logits.view(-1, 1, self.H, self.W)  # Reshape to single-channel image dimensions
        return logits


class AE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def loss(self, x: torch.tensor):
        x_recon = self.forward(x)
        return nn.functional.mse_loss(x_recon, x)