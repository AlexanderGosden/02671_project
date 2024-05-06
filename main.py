import torch
from src.models.AE import Encoder, Decoder, AE
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np
from src.utils.misc import load_config, animate_video, interpolate_linear
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from src.utils.build_model import build_model
from src.data_utils.boximage import boximage

# Set plotting
plotting = True

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

# Save videos
np.save('data/size_video.npy', size_video)
np.save('data/moving_video.npy', moving_video)

# Set device
device = ('cuda' if torch.cuda.is_available() else 'cpu')


for video, video_name in zip([size_video, moving_video], ['Regular', 'Moving']):

    config_path = f'configs/config_{video_name}.yaml'

    CFG = load_config(config_path)

    # convert data
    video = torch.tensor(video.astype(np.float32))
    train_loader = DataLoader(dataset=video, batch_size=CFG['training']['batch_size'], shuffle=True)



    # Loop over models
    model_names = ['Regular', 'Interp']
    for model_name in model_names:
        print(f'Training {model_name} model on {video_name} video')


        # build the model
        model = build_model(CFG, device, loss_type=model_name) 
        

        # Train the model
        trainer = Trainer(model, train_loader, config_path = config_path, device = device)
        losses = trainer.train()

        # Save trained model
        torch.save(model.state_dict(), f = f'data/model_weights/{model_name}_{video_name}.pth')

            
        # Print some statistics
        print(f'Minimal loss: {np.min(losses)}')
        output = model(video).detach().cpu()
        print(f'Reconstruction of pure video loss: {np.mean(np.square((output-video).numpy()))}')


        # Plot results
        if plotting:
            # Plot and save loss history
            plt.semilogy(losses)
            plt.savefig(f'figures/loss_{model_name}_{video_name}.png', dpi=300)
            plt.show()

            # Plot reconstructed video
            ani = animate_video(output)
            plt.show()

            # Interpolate in latent space
            encoded = model.encoder(video.to(device)).detach().cpu().numpy()
            encoded_interpolated = interpolate_linear(np.arange(len(encoded)), np.arange(0, len(encoded), 0.25), encoded).astype(np.float32)
            output_interpolated = model.decoder(torch.tensor(encoded_interpolated).to(device)).detach().cpu().numpy()
            # Plot interpolated video
            ani = animate_video(output_interpolated)
            plt.show()
            
            # Interpolate in pixel space
            interpolated_video = interpolate_linear(np.arange(len(video)), np.arange(0, len(video), 0.25), video)
            # Plot interpolated video
            ani = animate_video(interpolated_video)
            plt.show()
