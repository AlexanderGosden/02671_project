import torch
from src.models.AE import Encoder, Decoder, AE
from src.utils.build_model import build_model, build_model_interp
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
from src.utils.misc import load_config, interpolate_linear
from src.data_utils.boximage import boximage
import numpy as np
from matplotlib import pyplot as plt


n_frames = 60
box_widths = np.hstack([np.linspace(0, 100, n_frames//4), np.linspace(100, 10, n_frames//4), np.linspace(10, 90, n_frames//4), np.linspace(90, 0, n_frames//4)])
box_heights = box_widths
center_x = np.hstack([np.linspace(0, 100, n_frames // 2), np.linspace(100, 0, n_frames // 2)])
center_y = np.linspace(0, 127, n_frames)
size_video = np.array([boximage(box_width, box_height) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])
moving_video = np.array([boximage(box_width, box_height, center = (cx, cy)) for box_width, box_height, cx, cy in zip(box_widths, box_heights, center_x, center_y)])

# Set seed for reproducibility
np.random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



for video, video_name in zip([size_video, moving_video], ['Regular', 'Moving']):
    interpolated_video = interpolate_linear(np.arange(len(video)), np.arange(0, len(video), 0.1), video)
    # Load configuration
    CFG = load_config(f'configs/config_{video_name}.yaml')

    training_data = list(video.astype(np.float32))
    train_loader = DataLoader(dataset=training_data, batch_size=CFG['training']['batch_size'], shuffle=True)

    models = [build_model('AutoEncoder', CFG, device), build_model_interp('AutoEncoder', CFG, device)]
    model_names = ['Regular', 'Interp']

    data = video.astype(np.float32)
    data = list(DataLoader(dataset=data, batch_size=len(data), shuffle=False))[0]


    frames_of_interest = [55, 23]
    frame_names = ['Successful', 'Unsuccessful']
    fig, axs = plt.subplots(len(frames_of_interest), 3, figsize=(3.5, 3.5))
    j = 1
    for model, model_name in zip(models, model_names):
        model_path = "data/model_weights/"+model_name+"_"+video_name+".pth"

        # Load the saved model weights
        model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode
        model.eval()

        encoded = model.encoder(data.to(device)).detach().cpu().numpy()
        encoded_interpolated = interpolate_linear(np.arange(len(encoded)), np.arange(0, len(encoded), 0.1), encoded).astype(np.float32)
        output_interpolated = model.decoder(torch.tensor(encoded_interpolated).to(device)).detach().cpu().numpy()
        for i, frame in enumerate(frames_of_interest):
            axs[i, j].imshow(output_interpolated[frame], cmap='gray', vmin = 0, vmax = 255)
        j += 1

    for i, frame in enumerate(frames_of_interest):
        axs[i, 0].imshow(interpolated_video[frame], cmap='gray', vmin=0, vmax=255)
        axs[i, 0].set_title(frame_names[i])

        # hide axes labels
        for j in range(2):
            axs[i,j].axes.get_xaxis().set_visible(False)
            axs[i,j].set_yticklabels([])
            axs[i,j].set_yticks([])

        axs[0, 0].set_ylabel('Naive interpolation')
        axs[0, 1].set_ylabel('Encoded interpolation')
        axs[0, 2].set_ylabel('Encoded interpolation w. loss')
        fig.tight_layout()