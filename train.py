import argparse
import torch
from src.model import VQVAE
from src.dataset import get_cifar10_loaders
from src.trainer import VQVAETrainer
from src.utils import count_parameters

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on CIFAR-10")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs") 
    parser.add_argument("--num_hiddens", type=int, default=128, help="Number of hidden channels")
    parser.add_argument("--num_residual_hiddens", type=int, default=32, help="Number of residual hidden channels")
    parser.add_argument("--num_residual_layers", type=int, default=2, help="Number of residual layers")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of embeddings in codebook")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Size of codebook")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost beta")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    print("Initializing Data Loaders...")
    train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)
    
    print("Initializing VQ-VAE Model...")
    model = VQVAE(num_hiddens=args.num_hiddens, 
                  num_residual_layers=args.num_residual_layers,
                  num_residual_hiddens=args.num_residual_hiddens,
                  num_embeddings=args.num_embeddings,
                  embedding_dim=args.embedding_dim,
                  commitment_cost=args.commitment_cost)
    
    print(f"Model Parameters: {count_parameters(model):,}")
    
    trainer = VQVAETrainer(model, train_loader, val_loader, 
                           learning_rate=args.learning_rate, 
                           device=args.device)
    
    print("Starting Training...")
    trainer.fit(args.epochs)

if __name__ == "__main__":
    main()
