from src.utils.misc import load_config
from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
class Trainer:
    def __init__(self, model, train_loader, config_path: str, device: str) -> None:
        self.model = model
        self.train_loader = train_loader
        self.CFG = load_config(config_path)
        self.device=device
        self.__initialize_training_utils()
    
    def __initialize_training_utils(self):
        optimizer_name = self.CFG['training']['optimization']['optimizer_name']
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.CFG['training']['optimization']['Adam']['lr'],
                                        betas=self.CFG['training']['optimization']['Adam']['betas'])
    def train(self):
        self.model.train()
        self.model.to(self.device)
        total_losses = []
        best_loss = np.inf
        with tqdm(total=self.CFG['training']['n_epochs']*len(self.train_loader), desc="Training", unit="iter") as pbar:
            for _ in range(self.CFG['training']['n_epochs']):
                losses = []
                for batch in self.train_loader:
                    batch.to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model.loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    pbar.set_postfix({'Current loss': sum(losses)/len(losses)}, refresh=True)
                    pbar.update(1)
                avg_loss = sum(losses)/len(losses)
                # Save best model
                if avg_loss < best_loss:
                    best_model = self.model.state_dict()
                total_losses.append(avg_loss)
        # Load the model with the lowest loss
        self.model.load_state_dict(best_model)  
        return total_losses
