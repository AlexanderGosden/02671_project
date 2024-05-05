import argparse
import os
import torch
from src.models.AE import Encoder, Decoder, AE
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np
from src.utils.misc import load_config, animate_video, interpolate_linear
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from src.utils.build_model import build_model, build_model_interp
from src.data_utils.boximage import boximage

# Set seed
np.random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)

# Make videos
n_frames = 60
box_widths = np.hstack([np.linspace(0, 100, n_frames//4), np.linspace(100, 10, n_frames//4), np.linspace(10, 90, n_frames//4), np.linspace(90, 0, n_frames//4)])
box_heights = box_widths
center_x = np.hstack([np.linspace(0, 100, n_frames // 2), np.linspace(100, 0, n_frames // 2)])
center_y = np.linspace(0, 127, n_frames)
size_video = np.array([boximage(box_width, box_height) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])
moving_video = np.array([boximage(box_width, box_height, center = (cx, cy)) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])

# Set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')


for video, video_name in zip([size_video, moving_video], ['Regular', 'Moving']):
    interpolated_video = interpolate_linear(np.arange(len(video)), np.arange(0, len(video), 0.25), video)
    CFG = load_config(f'configs/config_{video_name}.yaml')

    training_data = list(video.astype(np.float32))
    train_loader = DataLoader(dataset=training_data, batch_size=CFG['training']['batch_size'], shuffle=True)
    data = video.astype(np.float32)
    data = list(DataLoader(dataset=data, batch_size=len(data), shuffle=False))[0]

    models = [build_model('AutoEncoder', CFG, device), build_model_interp('AutoEncoder', CFG, device)]
    model_names = ['Regular', 'Interp']


    for model, model_name in zip(models, model_names):
        trainer = Trainer(model, train_loader, config_path = f'configs/config_{video_name}.yaml', device = device)
        losses = trainer.train()

        # Save model
        torch.save(model.state_dict(), f = f'data/model_weights/{model_name}_{video_name}.pth')

        plt.semilogy(losses)
        plt.savefig(f'figures/loss_{model_name}_{video_name}.png', dpi=300)
        plt.show()
        print(f'Minimal loss: {np.min(losses)}')

        output = model(data.to(device)).detach().cpu().numpy()

        print(f'Reconstruction of pure video loss: {np.mean(np.square(output - data.numpy()))}')
        ani = animate_video(output)
        plt.show()

        encoded = model.encoder(data.to(device)).detach().cpu().numpy()
        encoded_interpolated = interpolate_linear(np.arange(len(encoded)), np.arange(0, len(encoded), 0.25), encoded).astype(np.float32)
        output_interpolated = model.decoder(torch.tensor(encoded_interpolated).to(device)).detach().cpu().numpy()

        ani = animate_video(output_interpolated)
        plt.show()
        ani = animate_video(interpolated_video)
        plt.show()
