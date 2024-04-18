import argparse
import os
import torch
from src.models.AE import Encoder, Decoder, AE
from src.training.trainer import Trainer
from torch.utils.data import DataLoader
import numpy as np
from src.utils.misc import load_config, animate_video
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def build_model(model_type: str, CFG, device: str):
    H = CFG['data']['H']
    W = CFG['data']['W']
    latent_dim = CFG['AE']['D']
    if model_type == 'AutoEncoder':
        encoder = Encoder(H, W, latent_dim)
        decoder = Decoder(H, W, latent_dim)
        model = AE(encoder, decoder)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

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
            model.eval()
            print('Model loaded.')
        else:
            raise ValueError(f"Model weights not found at {path}")

    
        # Evaluate model
        test_data = np.load('data/boxvideo.npy').astype(np.float32)
        
        with torch.no_grad():
            output = model(test_data)
                
        # Animate the output
        ani = animate_video(output)
        plt.show()







