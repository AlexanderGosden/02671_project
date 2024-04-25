import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, H: int, W: int, latent_dim: int, N_latent_1 : int, N_latent_2 : int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.latent_dim = latent_dim
        self.N_latent_1 = N_latent_1
        self.N_latent_2 = N_latent_2
        self.encoder_net = self.__get_encoder_net_conv()

    def __get_encoder_net(self):
        encoder_net = nn.Sequential(

            nn.Flatten(),  # flatten image [N x H*W]
            nn.Linear(self.H * self.W, self.N_latent_1),
            nn.ReLU(),
            nn.Linear(self.N_latent_1, self.N_latent_2),
            nn.ReLU(),
            nn.Linear(self.N_latent_2, self.latent_dim) 
        )
        return encoder_net

    
    def __get_encoder_net_conv(self): #seems to be very slow, so don't use
        padding = (1,1)
        stride = (1,1)
        kernel_size = (3, 3)
        dilation = (1,1)

        C_out = 8

        H_out = (self.H + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
        W_out = (self.W + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1

        encoder_net = nn.Sequential(
            nn.Conv2d(1, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Flatten(),  # flatten the output from the conv layers
            nn.Linear(C_out * H_out * W_out, self.N_latent_2),
            nn.ReLU(),
            nn.Linear(self.N_latent_2, self.latent_dim)
        )
        return encoder_net

    def forward(self, x: torch.tensor):
        x = x.view(-1, 1, self.H, self.W)  # Reshape to single-channel image dimensions. Torch expects [N x C x H x W] where C is channel
        return self.encoder_net(x)


class Decoder(nn.Module):
    def __init__(self, H: int, W: int, latent_dim: int, N_latent_1 : int, N_latent_2 : int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.latent_dim = latent_dim
        self.N_latent_1 = N_latent_1
        self.N_latent_2 = N_latent_2
        self.decoder_net = self.__get_decoder_net()
        

    def __get_decoder_net(self):
        decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.N_latent_2),
            nn.ReLU(),
            nn.Linear(self.N_latent_2, self.N_latent_1),
            nn.ReLU(),
            nn.Linear(self.N_latent_1, self.H * self.W), 
        )
        return decoder_net
    

    def forward(self, z: torch.tensor):
        logits = self.decoder_net(z)  # [N x H*W]
        logits = logits.view(-1, self.H, self.W)  # Reshape to single-channel image dimensions
        return logits


class AE(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.tensor):
        x = x
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def loss(self, x: torch.tensor):
        x_recon = self.forward(x)
        return nn.functional.mse_loss(x_recon, x)