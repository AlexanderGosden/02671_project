import argparse
import os
import torch


def build_model(model_type: str, device: str):
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='AutoEncoder', choices=['AutoEncoder'], help='What type of model to use (default: %(default)s)')
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    if args.mode == 'train':
       print(f'Training...')
        
    if args.mode == 'eval':
        print(f'Evaluating...')


    if args.mode == 'sample':
        model = build_model(args.model_type, device)
        
    