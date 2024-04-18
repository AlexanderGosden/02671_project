import argparse
import os
import torch
from src.models.AE import Encoder, Decoder, AE
from src.training.trainer import Trainer
from torch.utils.data import DataLoader

def build_model(model_type: str, device: str, H: int, W: int, latent_dim: int):
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
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    if args.mode == 'train':
        print(f'Training...')
        # Build model
        model = build_model(args.model_type, device, H=200, W=200, latent_dim=32)
        # Dummy data loader (replace this with your actual data loader)
        train_loader = DataLoader(dataset=[], batch_size=args.batch_size, shuffle=True)
        # Initialize trainer
        trainer = Trainer(model, train_loader, config_path='config.yaml', device=device)
        # Start training
        losses = trainer.train()
        print("Training finished.")
        
    if args.mode == 'eval':
        print(f'Evaluating...')