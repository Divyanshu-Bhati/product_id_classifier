import os
import json
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Use the non-GUI backend for saving plots, disable to show plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.parse_data import DataCreator
from core.classifier.classifier_head import ClassifierHead
from core.vae.custom_vae_model import VAE

import wandb
import warnings
warnings.filterwarnings("ignore", message=".*Profiler clears events at the end of each cycle.*") # Ignore profiler memory flush warnings

class Inference:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        with open("utils/configs.json", "r") as f:
            self.config = json.load(f)
        self.training_history = self.config["training_history"]
        self.seed = self.config["random_seed"]
        self.input_path = self.config["inputs_path"]
        self.data_filters = self.config["data_filters"]
        try:
            with open(os.path.join(self.training_history, "vocab.json"), "r") as f:
                vocab_data = json.load(f)
            scaler_data = torch.load(os.path.join(self.training_history, "cls_scaler_params.pth"), map_location=self.device)
            self.mean = scaler_data['mean']
            self.std = scaler_data['std']
        except:
            raise Exception("Vocabulary and scaler params files not found. Please re-run both the training scripts (VAE -> CLS).")
        
        self.char2idx = vocab_data['char2idx']
        self.idx2char = {int(k): v for k, v in vocab_data['idx2char'].items()}
        self.train_max_length = vocab_data['max_length']

        # Initialize data creator
        self.data_creator = DataCreator(
            input_path=self.input_path,
            training_history=self.training_history,
            seed=self.seed
        )
        # Initialize and load the VAE
        self.vae = self._load_vae()
        # Initialize and load the classifier
        self.classifier = self._load_classifier()

    def _load_vae(self):
        hp = self.config["vae_configs"]["hyperparameters"]
        model = VAE(
            vocab_size=len(self.char2idx),
            embedding_dim=hp["embedding_dim"],
            hidden_dim=hp["hidden_dim"],
            latent_dim=hp["latent_dim"],
            max_length=self.train_max_length
        ).to(self.device)
        
        path = os.path.join(self.training_history, "weights", "best_weights", "vae_best_weight.pth")
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _load_classifier(self):
        model = ClassifierHead(input_dim=133).to(self.device)
        path = os.path.join(self.training_history, "weights", "best_weights", "cls_best_weight.pth")
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model
    
    def extract_features_inference(self, X_data, vae_model):
        """
        Uses the pretrained VAE to extract the following five scores (one per sample):
         1. recon_error: average cross-entropy loss.
         2. kl_loss: average KL divergence.
         3. z_mean: mean of latent space.
         4. z_log_var: log variance of latent space.
         5. char_ratio: fraction of alphabetic characters.
         6. num_ratio: fraction of numeric characters.
         7. spcl_char_ratio: fraction of special characters.
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
            torch.log1p(recon_loss_per_sample).unsqueeze(1), 
            torch.log1p(kl_loss).unsqueeze(1),
            z_mean,
            z_log_var,
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
    
    def filter_inference_list(self, data_list, data_filters):
        """
        Fast filtering for inference. 
        Returns a boolean mask where True means the string is 'valid enough' to be processed.
        """
        data_filters = self.config.get("data_filters", {})
        max_len = data_filters.get("max_len", 14)
        min_len = data_filters.get("min_len", 10)
        illegal_chars = pd.read_csv(os.path.join(self.input_path, "wrong_chars.csv"))['found_chars'].tolist()
        illegal_set = set(illegal_chars)

        def is_structurally_sound(s):
            s = str(s)
            if not (min_len <= len(s) <= max_len):
                return False
            if any(char in illegal_set for char in s):
                return False
            if any(ord(char) > 127 for char in s):
                return False
            return True

        mask = [is_structurally_sound(x) for x in data_list]
        return mask

    def batch_inference(self, data_list, batch_size=32):
        """
        Input: list of strings, batch_size
        Output: list of 'yes'/'no' predictions
        Note: Works for single inputs (data_list=['string']) or large batches. Set batch size to 32 default.
        """
        if isinstance(data_list, pd.Series):
            data_list = data_list.tolist()
        
        print("Running inference...")
        mask = self.filter_inference_list(data_list, self.data_filters)
        
        # Pass invalid inputs directly to csv with 'invalid' prediction
        process_indices = [i for i, is_valid in enumerate(mask) if is_valid]
        strings_to_process = [data_list[i] for i in process_indices]
        results_map = {i: "invalid" for i, is_valid in enumerate(mask) if not is_valid}
        
        if strings_to_process:
            score_vector, debug_info = self.extract_features_inference(strings_to_process, self.vae)
            score_vector = (score_vector - self.mean) / (self.std + 1e-7)
            ds = TensorDataset(score_vector)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            
            processed_preds = []
            profiler_activities = [torch.profiler.ProfilerActivity.CPU] # Automatically tracks MPS activity for MacOS
            if torch.cuda.is_available():
                profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=profiler_activities,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./training_history/profiler_logs'),
                record_shapes=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    for batch in tqdm(loader, desc=f"Inference"):
                        x = batch[0].to(self.device)
                        
                        logits = self.classifier(x).squeeze(-1)
                        if logits.dim() == 0: logits = logits.unsqueeze(0)
                        probs = torch.sigmoid(logits)
                        labels = ["yes" if p >= 0.5 else "no" for p in probs.cpu().numpy()]
                        processed_preds.extend(labels)
                        prof.step()
                        
                for idx, pred in zip(process_indices, processed_preds):
                    results_map[idx] = pred

        final_preds = [results_map[i] for i in range(len(data_list))]        
        df = pd.DataFrame({
            "input_id": data_list,
            "prediction": final_preds
        })

        output_dir = "tests/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "inference_results.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Inference complete. Results saved to {output_path}")
        return df
    
    def _save_local_report(self, cm, acc, prec, rec, f1, auc):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Invalid', 'Predicted Valid'],
                    yticklabels=['Actual Invalid', 'Actual Valid'])
        plt.title(f'Final Confusion Matrix\nAccuracy: {acc:.4f} | AUC: {auc:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        img_path = os.path.join("tests", "final_cm.png")
        plt.savefig(img_path)
        plt.close()

        txt_path = os.path.join("tests", "final_metrics_summary.txt")
        with open(txt_path, "w") as f:
            f.write("--- FINAL TEST EVALUATION ---\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"ROC-AUC:   {auc:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

        print(f"Visual report saved to {img_path}")
        print(f"Text summary saved to {txt_path}")
    
    def run_final_test(self):
        wandb_run = wandb.init(
            entity="divyanshubhati",
            project="PD_CLASSIFIER_TEST",
            config={
                "dataset": "kaggle/arhamrumi/amazon-product-reviews/",
                "architecture": "VAE->CLS",
                "notes": "Final test metrics logged on best pre-trained weights."
            }
        )

        print("Testing the whole system...")
        X_test, y_test = self.data_creator.run_test(data_filters=self.data_filters, task_type="test")
        results_df = self.batch_inference(X_test, batch_size=32)
        y_pred_numeric = [1 if p == 'yes' else 0 for p in results_df['prediction']]
        
        # Calculate metrics & save
        acc = accuracy_score(y_test, y_pred_numeric)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_numeric, average='binary')
        auc = roc_auc_score(y_test, y_pred_numeric) 
        cm = confusion_matrix(y_test, y_pred_numeric)
        results_df['ground_truth'] = y_test
        failures = results_df[(results_df['prediction'] == 'yes') & (results_df['ground_truth'] == 0)]
        failures.head(5).to_csv("tests/failure_modes_examples.csv", index=False)
        final_log = {
            "final_test_acc": acc,
            "final_test_f1": f1,
            "best_vae_weights": "vae_best_weight.pth",
            "best_cls_weights": "cls_best_weight.pth",
            "confusion_matrix": cm.tolist()
        }
        with open("tests/final_evaluation_log.json", "w") as f:
            json.dump(final_log, f, indent=4)

        # Visuals & W&B
        self._save_local_report(cm, acc, precision, recall, f1, auc)
        wandb_run.log({
            "test/accuracy": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/roc_auc": auc,
            "test/conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test,
                preds=y_pred_numeric,
                class_names=["Invalid", "Valid"]
            )
        })
        wandb_run.finish()
        print(f"Final Accuracy: {acc:.4f}. All cases saved to tests/ dir. Testing complete.")

if __name__ == "__main__":
    import pprint
    
    inf = Inference()
    try:
        # Begin custom inference
        custom_data_df = pd.read_csv(os.path.join("data", "custom_data.csv"))
        sample_data = custom_data_df["input_id"].tolist()
    except:
        # If no custom data provided, first run test pipeline and hit batch inference using a sample list
        # Begin test
        inf.run_final_test()
        sample_data = ["B001L6FPTI", "ABC-123456", "INVALID!!!@@@", "1234"]    
        
    predictions = inf.batch_inference(sample_data, batch_size=32)
    pprint.pprint(predictions.head())