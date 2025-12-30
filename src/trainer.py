import torch
import torch.optim as optim
import os
import time
from tqdm import tqdm
from .utils import show_img_grid

class VQVAETrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=1e-3, device="cpu", save_dir="results"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_recon_error = 0.0
        train_perplexity = 0.0
        
        start_time = time.time()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for i, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = torch.mean((data_recon - data)**2) / torch.var(data) # Normalized MSE
            
            loss = recon_error + vq_loss
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_recon_error += recon_error.item()
            train_perplexity += perplexity.item()
            
            if (i+1) % 100 == 0:
                 pbar.set_postfix({"Loss": loss.item(), "Recon": recon_error.item(), "Perplexity": perplexity.item()})
        
        avg_loss = train_loss / len(self.train_loader)
        avg_recon = train_recon_error / len(self.train_loader)
        avg_perp = train_perplexity / len(self.train_loader)
        
        print(f"Epoch {epoch} Done. Loss: {avg_loss:.4f}, ReconError: {avg_recon:.4f}, Perplexity: {avg_perp:.4f}, Time: {time.time()-start_time:.2f}s")
        
    def evaluate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_perp = 0.0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                vq_loss, data_recon, perplexity = self.model(data)
                
                recon_error = torch.mean((data_recon - data)**2) / torch.var(data)
                loss = recon_error + vq_loss
                
                val_loss += loss.item()
                val_recon += recon_error.item()
                val_perp += perplexity.item()
        
        avg_loss = val_loss / len(self.val_loader)
        print(f"Epoch {epoch} [Val] Loss: {avg_loss:.4f}, ReconError: {val_recon/len(self.val_loader):.4f}, Perplexity: {val_perp/len(self.val_loader):.4f}")
        
        # Visualize reconstruction
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)
        _, data_recon, _ = self.model(data)
        
        show_img_grid(data, title=f"Epoch {epoch} - Original", save_path=os.path.join(self.save_dir, f"epoch_{epoch}_orig.png"))
        show_img_grid(data_recon, title=f"Epoch {epoch} - Recon", save_path=os.path.join(self.save_dir, f"epoch_{epoch}_recon.png"))

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            self.evaluate(epoch)
            
            # Save model
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"vqvae_epoch_{epoch}.pth"))
