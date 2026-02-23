import os
import glob
import json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Enable cuDNN benchmark for consistent input sizes BEFORE loading the VAE
torch.backends.cudnn.benchmark = True

from core.vae.custom_vae_model import VAE
from utils.parse_data import DataCreator

import wandb
import warnings
warnings.filterwarnings("ignore", message=".*Profiler clears events at the end of each cycle.*") # Ignore profiler memory flush warnings

class TrainVAE:
    def __init__(self, experiment_mode=False):
        # Catch arguments
        self.experiment_mode = experiment_mode
        
        with open("utils/configs.json", "r") as f:
            config = json.load(f)
        
        # Configuration parameters
        self.input_path = config["inputs_path"]
        self.training_history = config["training_history"]
        vae_configs = config["vae_configs"]
        self.continue_training = vae_configs["continue_training"]
        self.early_stopping = vae_configs["early_stopping"]
        self.hyperparameters = vae_configs["hyperparameters"]
        self.random_seed = config["random_seed"]
        self.data_filters = config["data_filters"]
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Use DataCreator to load & process data (with train/val split)
        print("Loading and processing data...")
        (self.max_length,
         self.X_train_padded,
         self.X_val_padded,
         self.y_val,
         self.char2idx,
         self.idx2char) = DataCreator(input_path=self.input_path, training_history=self.training_history, seed=self.random_seed). \
                            run_vae(data_filters=self.data_filters, task_type="train_vae")

        # Set hyperparameters for model creation
        self.num_epochs = 10 if self.experiment_mode else vae_configs["epochs"]
        self.vocab_size = len(self.char2idx)
        self.batch_size = self.hyperparameters["batch_size"]
        self.embedding_dim = self.hyperparameters["embedding_dim"]
        self.latent_dim = self.hyperparameters["latent_dim"]
        self.hidden_dim = self.hyperparameters["hidden_dim"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.patience = self.hyperparameters["patience"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        
        # Initialize the VAE model.
        print("Initializing the VAE model...")
        self.vae_model = VAE(vocab_size=self.vocab_size,
                            embedding_dim=self.embedding_dim,
                            hidden_dim=self.hidden_dim,
                            latent_dim=self.latent_dim,
                            max_length=self.max_length
                        )
        self.vae_model.to(self.device)
        
        # Setup wandb for tracking runs
        self.wandb_run = wandb.init(
            entity="divyanshubhati",
            project="PD_VAE_PRE_TRAINING",
            config={
                "tags": ["experiment"] if self.experiment_mode else ["full_run"],
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "latent_dim": self.latent_dim,
                "epochs": self.num_epochs,
                "dataset": "kaggle/arhamrumi/amazon-product-reviews/",
                "architecture": "MLP-VAE",
                "notes": "Testing pipeline with sample data and epochs"
            }
        )
    
    def train_vae_model(self):
        print("Starting VAE pre-training...")

        # Create DataLoaders for training and validation.
        print("Creating DataLoaders...")
        train_tensor = torch.tensor(self.X_train_padded, dtype=torch.long)
        train_dataset = TensorDataset(train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True) # TODO pin memory

        val_x_tensor = torch.tensor(self.X_val_padded, dtype=torch.long)
        val_y_tensor = torch.tensor(self.y_val, dtype=torch.float)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
                            
        # Directories for saving weights and logs.
        training_weights_dir = os.path.join(self.training_history, "weights")  # root folder for storing training weights
        best_weights_dir = os.path.join(training_weights_dir, "best_weights")
        logs_dir = os.path.join(self.training_history, "local_logs")
        os.makedirs(training_weights_dir, exist_ok=True)
        os.makedirs(best_weights_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        best_val_loss = float('inf')
        start_epoch = 0
        if self.continue_training:
            files = [f for f in os.listdir(training_weights_dir)
                        if f.startswith("training_vae_weights_epoch_") and f.endswith(".pth")]
            if files:
                epochs_numbers = [int(f.split("_")[-1].split(".")[0]) for f in files]
                start_epoch = max(epochs_numbers)
                latest_weights_file = os.path.join(training_weights_dir, f"training_vae_weights_epoch_{start_epoch}.pth")
                self.vae_model.load_state_dict(torch.load(latest_weights_file, map_location=self.device))
                print(f"Continuing VAE training from epoch: {start_epoch}")

        # Using AdamW with L2 regularization (L2 weight decay, decoupled from gradient update) and set up a cosine annealing scheduler. # TODO
        optimizer = torch.optim.AdamW(self.vae_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.hyperparameters.get("l2_lambda", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, # Need schedular to deal with posterior collapse aka learning the whole space TODO
                                                                T_max=self.num_epochs,
                                                                eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)  # Ignores the padded token while calculating loss. In sum vs mean -> ELBO expects reconstruction loss to be summed over the logs, mean will introduce scale variance.
        # TODO: Add static kl_beta or a scheduler to about KL divergence and prevent it from overpowering the RL
        
        print("Beginning VAE training for", self.num_epochs, "epochs...")
        self.vae_model.train()
        no_improve_count = 0
        logs_list = []
        
        # Add PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], # Add metal for mac runs?
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./training_history/profiler_logs'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for epoch in range(start_epoch, self.num_epochs + start_epoch):
                epoch_loss = 0
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                print(f"\nEpoch {epoch+1}/{self.num_epochs + start_epoch}")
                for batch in tqdm(train_dataloader, desc="Training", leave=False):
                    x = batch[0].to(self.device)
                    optimizer.zero_grad()
                    recon_logits, z_mean, z_log_var = self.vae_model(x)
                    recon_logits = recon_logits.view(-1, self.vocab_size)
                    x_flat = x.view(-1)
                    recon_loss = criterion(recon_logits, x_flat) # Mean over all errors for each batch
                    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var)) # Mean over entire latent space
                    loss = (recon_loss + kl_loss) # / x.size(0) # Averaged over batch size for each batch, already taken mean

                    # L1 regularization. # TODO
                    l1_lambda = self.hyperparameters.get("l1_lambda", 0.0)
                    if l1_lambda > 0:
                        l1_norm = sum(p.abs().sum() for p in self.vae_model.parameters())
                        loss = loss + l1_lambda * l1_norm

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()

                num_batches = len(train_dataloader)
                avg_train_loss = epoch_loss / num_batches
                avg_train_recon = epoch_recon_loss / num_batches
                avg_train_kl = epoch_kl_loss / num_batches
                print(f"VAE Training Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f})")

                # Validation loop
                self.vae_model.eval()
                val_loss = 0
                total_val_samples = len(val_dataloader.dataset)
                # Tracking error by class (y=1 vs y=0)
                pos_errors = []
                neg_errors = []

                with torch.no_grad(): # no gradient calculation during validation
                    for batch in tqdm(val_dataloader, desc="Validation", leave=False):
                        x_val = batch[0].to(self.device)
                        y_val = batch[1].to(self.device)
                        recon_logits, mu, logvar = self.vae_model(x_val)
                        recon_loss = criterion(recon_logits.view(-1, self.vocab_size), x_val.view(-1))
                        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        val_loss += (recon_loss.item() + kl_loss.item()) # Accumulate total sum
                        per_sample_recon = F.cross_entropy(
                            recon_logits.transpose(1, 2), x_val, reduction='none', ignore_index=0
                        ).sum(dim=1)
                        
                        pos_errors.extend(per_sample_recon[y_val == 1].tolist())
                        neg_errors.extend(per_sample_recon[y_val == 0].tolist())
                avg_val_loss = val_loss / total_val_samples
                avg_pos_err = np.mean(pos_errors) if pos_errors else 0
                avg_neg_err = np.mean(neg_errors) if neg_errors else 0

                print(f"VAE Val Loss: {avg_val_loss:.4f}")
                print(f"Mean Recon Error (Valid IDs): {avg_pos_err:.4f}")
                print(f"Mean Recon Error (Negative): {avg_neg_err:.4f}")

                # Log epoch metrics.
                logs_list.append({
                    "epoch": epoch+1,
                    "train_loss": avg_train_loss,
                    "train_recon_loss": avg_train_recon,
                    "train_kl_loss": avg_train_kl,
                    "val_loss": avg_val_loss,
                    "val_pos_recon_error": avg_pos_err,
                    "val_neg_recon_error": avg_neg_err
                })

                # Remove existing training weight files (to keep only the latest).
                for file in glob.glob(os.path.join(training_weights_dir, "training_vae_weights_epoch_*.pth")):
                    os.remove(file)
                latest_weights_file = os.path.join(training_weights_dir, f"training_vae_weights_epoch_{epoch+1}.pth")
                torch.save(self.vae_model.state_dict(), latest_weights_file)

                # Save best weights if improved.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_weights_file = os.path.join(best_weights_dir, "vae_best_weight.pth")
                    torch.save(self.vae_model.state_dict(), best_weights_file)
                    print("Best weight updated with validation loss:", best_val_loss)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if self.early_stopping and no_improve_count >= self.patience:
                        print(f"Early stopping triggered. No improvement for {self.patience} epochs.")
                        break

                self.vae_model.train()
                scheduler.step()  # Update the learning rate -> cosine decay
                
                # Log to W&B for every epoch
                self.wandb_run.log({
                        "epoch": epoch+1,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "pos_recon_error": avg_pos_err,
                        "neg_recon_error": avg_neg_err,
                        "lr": optimizer.param_groups[0]['lr']
                    })
                prof.step() # Update the profiler

            # Save final weights and logs.
            df_logs = pd.DataFrame(logs_list)
            csv_path = os.path.join(logs_dir, "vae_training_metrics.csv")
            df_logs.to_csv(csv_path, index=False)
            print("Training metrics saved to:", csv_path)
            
        # Close the run
        print("VAE training complete.")
        self.wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store_true', help='Run small sample/fewer epochs')
    args = parser.parse_args()
    TrainVAE(experiment_mode=args.experiment).train_vae_model()