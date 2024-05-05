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


np.random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)


#n_frames_per_step = 8
#width_checkpoints = [0, 100, 10, 90, 0, 80, 10, 100, 10, 90, 0, 80, 10]
#height_checkpoints = [0, 100, 10, 90, 0, 80, 10, 80, 0, 90, 10, 100, 0]

#box_widths = np.hstack([np.linspace(width_checkpoints[i], width_checkpoints[i+1], n_frames_per_step) for i in range(len(width_checkpoints)-1)])
#box_heights = np.hstack([np.linspace(height_checkpoints[i], height_checkpoints[i+1], n_frames_per_step) for i in range(len(height_checkpoints)-1)])
#size_video = np.array([boximage(box_width, box_height) for box_width, box_height in zip(box_widths, box_heights)])


#n_frames_per_step = 12

#width_checkpoints = [0, 50, 10, 90, 0]
#height_checkpoints = width_checkpoints
#center_x_checkpoints = [20, 100, 80, 100, 20]
#center_y_checkpoint = [100, 20, 80, 100, 80]

#box_widths = np.hstack([np.linspace(width_checkpoints[i], width_checkpoints[i+1], n_frames_per_step) for i in range(len(width_checkpoints)-1)])
#box_heights = np.hstack([np.linspace(height_checkpoints[i], height_checkpoints[i+1], n_frames_per_step) for i in range(len(height_checkpoints)-1)])
#center_x = np.hstack([np.linspace(center_x_checkpoints[i], center_x_checkpoints[i+1], n_frames_per_step) for i in range(len(center_x_checkpoints)-1)])
#center_y = np.hstack([np.linspace(center_y_checkpoint[i], center_y_checkpoint[i+1], n_frames_per_step) for i in range(len(center_y_checkpoint)-1)])

#moving_video = np.array([boximage(box_width, box_height, center = (cx, cy)) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])

n_frames = 60
box_widths = np.hstack([np.linspace(0, 100, n_frames//4), np.linspace(100, 10, n_frames//4), np.linspace(10, 90, n_frames//4), np.linspace(90, 0, n_frames//4)])
box_heights = box_widths
center_x = np.hstack([np.linspace(0, 100, n_frames // 2), np.linspace(100, 0, n_frames // 2)])
center_y = np.linspace(0, 127, n_frames)
size_video = np.array([boximage(box_width, box_height) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])
moving_video = np.array([boximage(box_width, box_height, center = (cx, cy)) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])

#n_frames = 10
#box_widths = np.hstack([np.linspace(0, 100, n_frames), np.linspace(100, 10, n_frames)])
#box_heights = box_widths
#center_x = np.hstack([np.linspace(10, 80, n_frames), np.linspace(80, 60, n_frames)])
#center_y = np.hstack([np.linspace(80, 50, n_frames), np.linspace(50, 10, n_frames)])
#size_video = np.array([boximage(box_width, box_height) for box_width, box_height in zip(box_widths, box_heights)])
#moving_video = np.array([boximage(box_width, box_height, center = (cx, cy)) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])


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

        #save model
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

        #frames_of_interest = [55, 23]
        #frame_names = ['Successful', 'Unsuccessful']

        #fig, axs = plt.subplots(2, len(frames_of_interest), figsize=(3.5, 3.5))

        #for i, frame in enumerate(frames_of_interest):
        #    axs[0, i].imshow(interpolated_video[frame], cmap='gray', vmin=0, vmax=255)
        #    axs[0, i].set_title(frame_names[i])

        #    axs[1, i].imshow(output_interpolated[frame], cmap='gray', vmin = 0, vmax = 255)

            # hide axes labels
        #    for j in range(2):
        #        axs[j,i].axes.get_xaxis().set_visible(False)
        #        axs[j,i].set_yticklabels([])
        #        axs[j,i].set_yticks([])

        #axs[0, 0].set_ylabel('Naive interpolation')
        #axs[1, 0].set_ylabel('Encoded interpolation')

        #fig.tight_layout()

        #plt.savefig(f'figures/interpolation_comparison_{model_name}_{video_name}.png', dpi=300)



"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='AutoEncoder', choices=['AutoEncoder'], help='What type of model to use (default: %(default)s)')
    args = parser.parse_args()
    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    if args.mode == 'train':
        print(f'Training...')
        # Build model
        model = build_model(args.model_type,CFG,device)
        # Data loader
        training_data = list(np.load('data/boxvideo.npy').astype(np.float32))
        train_loader = DataLoader(dataset=training_data, batch_size=CFG['training']['batch_size'], shuffle=True)
        # Initialize trainer
        trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device)
        # Start training
        losses = trainer.train()
        print("Training finished.")

        # Save model
        torch.save(model.state_dict(), f = f'data/model_weights/{args.model_type}_weights.pt')
        
    if args.mode == 'eval':
        print(f'Evaluating...')

        model = build_model(args.model_type,CFG,device)
        path = f'data/model_weights/{args.model_type}_weights.pt'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            model.to(device)
            model.eval()
            print('Model loaded.')
        else:
            raise ValueError(f"Model weights not found at {path}")

    
        # Load data
        data = np.load('data/boxvideo.npy').astype(np.float32)
        data = list(DataLoader(dataset=data, batch_size=len(data), shuffle=False))[0]

        # Forward pass
        output = model(data.to(device)).detach().cpu().numpy()
                
        # Animate the output
        ani = animate_video(output)
        plt.show()
"""