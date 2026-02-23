import os
import json
from tqdm import tqdm
import argparse
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Enable cuDNN benchmark for consistent input sizes BEFORE loading the VAE
torch.backends.cudnn.benchmark = True

from utils.parse_data import DataCreator
from core.vae.custom_vae_model import VAE
from core.classifier.classifier_head import ClassifierHead

import wandb
import warnings
warnings.filterwarnings("ignore", message=".*Profiler clears events at the end of each cycle.*") # Ignore profiler memory flush warnings

class TrainCLS:
    def __init__(self, experiment_mode=False):
        self.experiment_mode = experiment_mode
        
        with open("utils/configs.json", "r") as f:
            config = json.load(f)
        self.input_path = config["inputs_path"]
        self.random_seed = config["random_seed"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = config["training_history"]
        self.data_filters = config["data_filters"]
        
        # Classifier hyperparameters
        cls_configs = config["cls_configs"]
        self.epochs = 10 if self.experiment_mode else cls_configs["epochs"]
        self.num_workers = 2 if self.experiment_mode else cls_configs["num_workers"]
        self.learning_rate = cls_configs["hyperparameters"]["learning_rate"]
        self.batch_size = 512 if self.experiment_mode else cls_configs["hyperparameters"]["batch_size"]
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Initialize DataCreator
        self.data_creator = DataCreator(input_path=self.input_path,
                                        training_history=self.training_history,
                                        seed=self.random_seed,
                                        experiment_mode=self.experiment_mode)
        
        # Load fixed vocabulary and training max length.
        self.train_max_length, self.char2idx, self.idx2char = self.data_creator.load_training_vocab()
        vae_configs = config["vae_configs"]
        self.vocab_size = len(self.char2idx)
        # Load VAE model with pre-trained weights
        self.vae_model = VAE(vocab_size=self.vocab_size,
                            embedding_dim=vae_configs["hyperparameters"]["embedding_dim"],
                            hidden_dim=vae_configs["hyperparameters"]["hidden_dim"],
                            latent_dim=vae_configs["hyperparameters"]["latent_dim"],
                            max_length=self.train_max_length
                        )
        self.vae_model.to(self.device)
        best_weights_file = os.path.join(self.training_history, "weights", "best_weights", "vae_best_weight.pth")
        print("Loading VAE weights from file:", best_weights_file)
        self.vae_model.load_state_dict(torch.load(best_weights_file, map_location=self.device))
        
        # Setup wandb for tracking runs
        self.wandb_run = wandb.init(
            entity="divyanshubhati",
            project="PD_CLS_TRAINING",
            config={
                "tags": ["experiment"] if self.experiment_mode else ["full_run"],
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "dataset": "kaggle/arhamrumi/amazon-product-reviews/",
                "architecture": "MLP-BCE-CLS",
                "notes": "Testing pipeline with sample data and epochs"
            }
        )

    def extract_features(self, X_data, vae_model):
        """
        Uses the pretrained VAE to extract the following five scores (one per sample):
         1. recon_error: average cross-entropy loss.
         2. kl_loss: average KL divergence.
         3. char_ratio: fraction of alphabetic characters.
         4. num_ratio: fraction of numeric characters.
         5. spcl_char_ratio: fraction of special characters.
        Also returns for debugging & error analysis:
         - hybrid_score = recon_error + kl_loss.
         - reconstructions: list of decoded strings.
         - idx2char dictionary.
        """
        all_padded = self.data_creator.pad_truncate_data(
            data_list=X_data, 
            fixed_max_length=self.train_max_length, 
            char2idx=self.char2idx
        )
        data_tensor = torch.tensor(all_padded, dtype=torch.long).to(self.device)

        # Forward pass VAE in eval mode
        vae_model.eval()
        with torch.no_grad():
            recon_logits, z_mean, z_log_var = vae_model(data_tensor)
            criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=0) # 'none' strategy gets per sample loss
            recon_loss_all = criterion(recon_logits.transpose(1, 2), data_tensor) # (B, L)
            actual_lengths = (data_tensor != 0).sum(dim=1).clamp(min=1)
            recon_loss_per_sample = recon_loss_all.sum(dim=1) / actual_lengths
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), dim=1)
            
        # Get input level features as signals
        additional_features = []
        for s in X_data:
            s = str(s).strip()
            L = len(s) if len(s) > 0 else 1
            char_count = sum(1 for c in s if c.isalpha())
            num_count = sum(1 for c in s if c.isdigit())
            spcl_count = L - (char_count + num_count)
            additional_features.append([char_count / L, num_count / L, spcl_count / L])
        additional_features = torch.tensor(additional_features, dtype=torch.float32).to(self.device)

        # Assemble all scores in a vector (used as input for ClassifierHead)
        score_vector = torch.cat([
            recon_loss_per_sample.unsqueeze(1),
            kl_loss.unsqueeze(1),
            additional_features
        ], dim=1)

        reconstructions = [
            self.data_creator.decode_sequence(sample, self.idx2char) 
            for sample in data_tensor.cpu().numpy()
        ]
        
        debug = {
            "hybrid_score": recon_loss_per_sample + kl_loss, # ELBO score
            "reconstructions": reconstructions
        }
        return score_vector, debug

    def display_confusion_matrix(self, cm, labels):
        console = Console()
        table = Table(title="Confusion Matrix")
        print(f"Confusion matrix evaluation on {cm.sum()} samples")
        table.add_column("", justify="center", style="bold")
        for label in labels:
            table.add_column(label, justify="center", style="bold")
        for i, row in enumerate(cm):
            row_values = [str(val) for val in row]
            table.add_row(labels[i], *row_values)
        console.print(table)
        
    def train_cls_model(self):
        """
        Train the classifier head using the 5 extracted scores.
        Labels: "yes" -> 1, "no" -> 0.
        """
        print("Running VAE-based classifier pipeline...")
        
        # Load datasets
        X_train, X_eval, y_train, y_eval = self.data_creator.run_cls(data_filters=self.data_filters, task_type="train_cls")
        
        print("Extracting features for classifier training...")
        train_score_vector, train_debug = self.extract_features(X_train, self.vae_model)
        eval_score_vector, eval_debug = self.extract_features(X_eval, self.vae_model)
        train_labels = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        eval_labels = torch.tensor(y_eval, dtype=torch.float32).to(self.device)
        
        # Standardize scores before training
        mean = train_score_vector.mean(dim=0)
        std = train_score_vector.std(dim=0) + 1e-7  # Small epsilon to prevent div by zero (common feature in a batch -> std. deviation 0 -> NaN in tensors)
        
        # Save training mean and std for inference
        torch.save({'mean': mean, 'std': std},os.path.join(self.training_history, "cls_scaler_params.pth")) 
        
        train_score_vector = (train_score_vector - mean) / std
        eval_score_vector = (eval_score_vector - mean) / std
        train_ds = TensorDataset(train_score_vector, train_labels)
        eval_ds = TensorDataset(eval_score_vector, eval_labels)
        train_dataloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        eval_dataloader = DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        # Freeze the VAE
        for param in self.vae_model.parameters():
            param.requires_grad = False
        
        # Set up classifier
        classifier_model = ClassifierHead(input_dim=train_score_vector.size(1)).to(self.device)  # input dimension of ClassifierHead should match the score dimensions; fails if losses have a large difference -> standardized in above step
        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.BCELoss()
        l1_lambda = 1e-5 # Strength of L1 regularization
        best_val_loss = float('inf')
        cls_weights_dir = os.path.join(self.training_history, "weights")
        local_log_path = os.path.join(self.training_history, "local_logs", "cls_training_metrics.csv")
        os.makedirs(cls_weights_dir, exist_ok=True)
        os.makedirs(os.path.dirname(local_log_path), exist_ok=True)
        csv_headers = ["epoch", "train_loss", "val_loss", "val_acc"]
        
        with open(local_log_path, "w") as f:
            f.write(",".join(csv_headers) + "\n")

        print("Training classifier head...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./training_history/profiler_logs'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            for epoch in range(self.epochs):
                classifier_model.train()
                epoch_loss = 0
                
                for batch in tqdm(train_dataloader, desc=f"Training", leave=False):
                    x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
                    optimizer.zero_grad()
                    outputs = classifier_model(x_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    
                    # L1 Regularization
                    l1_norm = sum(p.abs().sum() for p in classifier_model.parameters())
                    loss = loss + l1_lambda * l1_norm

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    prof.step()

                # Evaluation step
                classifier_model.eval()
                val_loss, correct, total = 0, 0, 0
                all_preds, all_labels = [], []
                
                with torch.no_grad():
                    for x_val, y_val in eval_dataloader:
                        val_outputs = classifier_model(x_val).squeeze()
                        val_loss += criterion(val_outputs, y_val).item()
                        
                        preds = (val_outputs > 0.5).float()
                        correct += (preds == y_val).sum().item()
                        total += y_val.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y_val.cpu().numpy())

                avg_train_loss = epoch_loss / len(train_dataloader)
                avg_val_loss = val_loss / len(eval_dataloader)
                val_acc = (correct / total) * 100

                # Log to CSV
                with open(local_log_path, "a") as f:
                    f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.2f}\n")

                # Log to W&B
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc
                }
                
                # Add CM to W&B on final epoch
                if epoch == self.epochs - 1:
                    log_dict["conf_mat"] = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels,
                        preds=all_preds,
                        class_names=["Invalid", "Valid"]
                    )
                self.wandb_run.log(log_dict)

                print(f"Epoch {epoch+1}: Loss: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_path = os.path.join(cls_weights_dir, "best_weights", "cls_best_weight.pth")
                    torch.save(classifier_model.state_dict(), best_path)
                    
            # Evaluation metrics
            best_path = os.path.join(cls_weights_dir, "best_weights", "cls_best_weight.pth")
            classifier_model.load_state_dict(torch.load(best_path))
            classifier_model.eval()

            # Load best weight (instead of last weight)
            best_path = os.path.join(cls_weights_dir, "best_weights", "cls_best_weight.pth")
            classifier_model.load_state_dict(torch.load(best_path))
            classifier_model.eval()
            
            all_probs, all_preds, all_labels_list = [], [], []
            with torch.no_grad():
                for x_val, y_val in eval_dataloader:
                    probs = classifier_model(x_val).squeeze()
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend((probs > 0.5).float().cpu().numpy())
                    all_labels_list.extend(y_val.cpu().numpy())

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels_list, all_preds, average='binary')
            roc_auc = roc_auc_score(all_labels_list, all_probs)

            print("\n" + "="*40)
            print(f"{'FINAL CLASSIFIER PERFORMANCE':^40}")
            print("="*40)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            print(f"Precision:     {precision:.4f}")
            print(f"Recall:        {recall:.4f}")
            print(f"F1 Score:      {f1:.4f}")
            print("-" * 40)

            # Feature Importance (Weight Inspection): the first layer (fc1) weights represents sparsity from L1 regularization
            first_layer_weights = classifier_model.fc1.weight.data
            feature_importance = torch.mean(torch.abs(first_layer_weights), dim=0)
            feature_names = ["Recon Error", "KL Loss", "Char Ratio", "Num Ratio", "Special Ratio"]
            
            print("FEATURE IMPORTANCE (Avg Abs Weight):")
            for name, weight in zip(feature_names, feature_importance):
                status = " [ACTIVE]" if weight > 0.005 else " [SPARSE/ZEROED]" # Really small weights are being zeroed by L1 -> classifier ignores it
                print(f"{name:15}: {weight:.6f}{status}")
            print("="*40)

            self.wandb_run.log({
                "final/roc_auc": roc_auc,
                "final/precision": precision,
                "final/recall": recall,
                "final/f1": f1,
                "final/conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_preds,
                    class_names=["Fraud", "Valid"]
                )
            })
            cm = confusion_matrix(all_labels, all_preds)
            self.display_confusion_matrix(cm, ["Fraud (no)", "Valid (yes)"])

        print("Classifier training complete.")
        self.wandb_run.finish()
        
# Run the classifier pipeline.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', action='store_true', help='Run small sample/fewer epochs')
    args = parser.parse_args()
    TrainCLS(experiment_mode=args.experiment).train_cls_model()