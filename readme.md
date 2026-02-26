# Product ID Anomaly Detection: A Hybrid VAE-Classifier Approach

Click the icon to Quick Start (in Google Colab): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Divyanshu-Bhati/product_id_classifier/blob/main/notebooks/colab_run.ipynb)

## 1. Problem Statement & Business Use Case

In large-scale e-commerce and logistics datasets, "Product IDs" often follow strict but undocumented structural rules. Traditional regex-based validation is brittle and fails to catch entries that "look" like IDs but violate latent structural patterns. This becomes especially problematic when factoring in unstructured or semi-structured data from the web, product documents, and spreadsheets across multiple organizations.

Drawing inspiration from VAE applications in fraud detection algorithms, I designed a classifier that uses the VAE to learn latent representations of "valid" IDs, and anything outside of that latent space is considered "invalid". This decision boundary is created by a simple MLP classifier that takes inputs from the VAE reconstruction. This approach addresses the extreme class imbalance in the industry: "valid" product IDs are clearly labeled and readily available, while "invalid" ones are highly varied and difficult to comprehensively define. This project, then, treats ID validation as a signal reconstruction task.

*Note: This repository contains a complete implementation of a VAE-Classifier originally trained on proprietary data, outputs of which were used to extract millions of products from unstructured PDFs. For this open-source version, I used a public Kaggle dataset. No proprietary data or weights are, or will be, shared.*

## 2. Getting Started
### Installation requirements: 
Python version 3.10, and CUDA version 12.6 setup is required (for GPU). GPU with at least 4GB of VRAM is recommended for full training.

### Quick Start:
The easiest way to set up and run the code is via the provided Google Colab link at the top of this document and simply run each cell.

### Local Setup:
You can find my best weights (trained on the Kaggle dataset), classifier scaler parameters, and training vocabulary in the [v1.0.0 Release](https://github.com/Divyanshu-Bhati/product_id_classifier/releases/tag/v1.0).
Download and store them in the following directories respectively:
* `training_history/vocab.json`
* `training_history/cls_scaler_params.pth`
* `training_history/weights/best_weights/vae_weights.pt`
* `training_history/weights/best_weights/cls_weights.pt`

*Note: Re-training the model will rewrite all above mentioned files, please create backups if needed.*

Then, run the following commands:
#### For Linux-based systems (WSL2, Ubuntu, etc.):
```bash
git clone https://github.com/Divyanshu-Bhati/product_id_classifier.git
cd product_id_classifier
source setup_linux.sh
python inference.py # only for testing (on test csv) using saved weights
python train_vae.py # train the vae first
python train_cls.py # then train the classifier
```

#### For MacOS:
```bash
git clone https://github.com/Divyanshu-Bhati/product_id_classifier.git
cd product_id_classifier
chmod +x setup_mac.sh
./setup_mac.sh
python3 inference.py # only for testing (on test csv) using saved weights 
python3 train_vae.py # train the vae first
python3 train_cls.py # then train the classifier
```

Optionally, you can use the `--experiment` flag to train the model (only applicable for training scripts) on a subset of the data and 10 epochs.

To run inference on a custom dataset, save a file named `custom_data.csv` in the `data/` directory and then run `python inference.py`. Please ensure the file contains a column named `input_id`. The outputs will be automatically saved in `tests/` with `input_id` and a corresponding `prediction` column.

*Note: If you are using Windows OS, it is best to activate WSL2 and run the setup_linux.sh script. For more on WSL, please refer to the official Microsoft documentation.* 

## 3. Directory Structure
```plaintext
├── core/
│   ├── vae/                     # VAE Architecture (custom_vae.py - model architecture, visualizer.py - experimental, unused in repo)
│   └── classifier/              # MLP Head for downstream decisioning
├── data/                        # Dataset splits (Train/Eval/Test) saved in .csv files
├── notebooks/
│   ├── colab_run.ipynb          # Google Colab notebook
│   ├── prepare_dataset.ipynb    # Preprocessing, EDA & initial split logic (with thought process and data decisions)
│   └── unit_test_notebook.ipynb # Integration tests & Determinism checks
├── utils/
│   ├── parse_data.py            # DataCreator class & preprocessing pipeline
│   └── configs.json             # Model training parameters and configurations
├── tests/                       # Store inference results (Gitignored)
├── training_history/            # Best weights, training vocabulary, local csv logs, profiler logs (Gitignored)
├── wandb/                       # Weights & Biases local logs (Gitignored)
├── inference.py                 # Main entry point for final testing and custom inference
├── train_cls.py                 # Classifier training & evaluation pipeline
├── train_vae.py                 # VAE training & evaluation pipeline
├── LICENSE
├── README.md                    # Project documentation
├── setup_linux.sh               # Automated .env creation and dependencies for WSL/Linux based systems
├── setup_mac.sh                 # Automated .env creation and dependencies for MacOS
├── requirements_linux.txt       # Project dependencies
└── requirements_mac.txt         # Project dependencies
```

## 4. Architecture
The system processes raw strings by converting characters into a fixed-length integer sequence (fixed vocabulary). The pipeline then operates in two distinct phases:

### Phase 1: Unsupervised Representation (VAE):
A Variational Autoencoder is trained exclusively on valid Product IDs to learn their underlying structural features. Because it only knows what a valid ID looks like, it struggles to reconstruct anomalies (negative sample reconstruction error increases as positive sample reconstruction error decreases). Crucially, the model's training is driven by the **Reconstruction Gap**. During validation, I track the difference between the reconstruction error of valid IDs and synthetic invalid IDs. The model saves its best weights only when this gap widens, thus ensuring it becomes highly accurate at reconstructing valid IDs while actively failing to reconstruct anomalies.

### Phase 2: Supervised Classification (MLP):
Instead of feeding raw text to a classifier, I use the trained VAE as a feature extractor. I pass a concatenated vector of deep and shallow signals to a Multi-Layer Perceptron, as described:
* **VAE Signals:** Reconstruction Loss (RL), KL Divergence (KL), Latent Mean ($z_\mu$), and Latent Variance ($z_{\log\sigma^2}$). *(Note: RL and KL are log-transformed to balance their scales).*
* **Heuristic Features:** Simple ratios of alphabetic, numeric, and special characters extracted from the raw string.
* The MLP applies L1 regularization during training. This acts as an automatic feature selector, weighting the most important signals and zeroing out the noise to output a final "Valid" or "Invalid" decision.

### The Loss Functions:
* For training the VAE, I minimize the $\beta$-Evidence Lower Bound (ELBO) with L1 Regularization, consisting of Reconstruction Cross-Entropy and KL Divergence. To prevent posterior collapse (a common issue in text/sequence VAEs where the decoder ignores the latent space) and to control the density of the latent clusters, a $\beta$ scaling factor to the KL Divergence is introduced, effectively making this a **$\beta$-VAE**. An L1 penalty is applied to the linear layers to encourage sparse feature representations.

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left[ \text{CE}(x_i, \hat{x}_i) - \frac{\beta}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_{i,j}^2) - \mu_{i,j}^2 - \sigma_{i,j}^2 \right) \right] + \lambda \sum_{w \in W} |w|$$

**Where:**
* $N$: The batch size (`self.batch_size`).
* $\text{CE}(x_i, \hat{x}_i)$: The Cross-Entropy reconstruction loss for the $i$-th sequence (summed over the sequence length, ignoring padding tokens).
* $\beta$: The `kl_beta` hyperparameter, scaling the importance of the KL Divergence penalty.
* $d$: The dimensionality of the latent space (`latent_dim`).
* $\mu_{i,j}$ and $\sigma_{i,j}^2$: The predicted latent mean and variance for the $j$-th dimension of the $i$-th sample.
* $\lambda$: The L1 regularization strength (`l1_lambda`).
* $W$: The subset of trainable weights (specifically linear layer weights) penalized by the L1 norm.

* For training the classifier, I apply **Binary Cross-Entropy with Logits** with L1 Regularization term, to ensure numerical stability, which mathematically fuses the sigmoid activation and the BCE calculation into a single layer. Additionally, I apply L1 Regularization to the network's weights (excluding biases and Batch Normalization parameters). This enforces sparsity, acting as an automatic feature selector that zero-outs unhelpful signals and forces the model to focus only on the most predictive latent features from the VAE.

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right] + \lambda \sum_{w \in W} |w|$$

**Where:**
* $N$: The batch size.
* $y_i \in \{0, 1\}$: The ground truth label for the $i$-th sample (1 for Valid, 0 for Invalid).
* $\hat{y}_i$: The raw logit prediction from the MLP before activation.
* $\sigma(\hat{y}_i)$: The sigmoid activation function $\frac{1}{1 + e^{-\hat{y}_i}}$.
* $\lambda$: The L1 regularization strength hyperparameter.
* $W$: The subset of trainable weights penalized by the L1 norm (excluding biases and Batch Normalization parameters to preserve activation scaling).

## 5. Data Preparation
* From the amazon reviews dataset (see section 8), I select the ProductID and UserID columns to be my valid IDs, and I create invalid IDs using the Time and Summary columns, and augment those randomly with character perturbations to act as real world noise.
* To mimic the problem of data imbalance, I use a majority of the data to train the VAE, and a smaller subset to train the classifier. The VAE (as mentioned above) was exclusively trained on valid IDs, and the negative samples (invalid IDs) were used only for evaluation. The classifier was trained on both valid and invalid IDs, and the VAE features were frozen during training.
* Detailed thought process, EDA, and the preprocessing pipeline can be found in `notebooks/prepare_dataset.ipynb`.

## 6. Experiments and Results
### Training Setup:
* VAE Phase: Trained using a Weighted Adam optimizer, and the learning rate was adjusted using a cosine learning rate scheduler. The beta parameter in the ELBO loss was tuned to encourage tighter clustering of valid ID representations without posterior collapse.
* Classifier Phase: The MLP head was trained on the frozen VAE features (Latent Mean, Variance, Reconstruction Loss and KL divergence) alongside heuristic string features.

### Evaluation Metrics:
* Given the extreme class imbalance (valid IDs heavily outnumbering invalid ones), standard accuracy is a flawed metric. The model was evaluated primarily on Precision, Recall, F1-Score, and ROC-AUC.

### Results:
* For the VAE: The best `Training Loss: 2.7671`, & `Val Loss: 27.6103`. The `Mean Reconstruction Error for Valid IDs was 0.0433`, and the `Mean Reconstruction Error for Invalid IDs was 52.5953`. The large gap between the reconstruction of Valid and Invalid IDs, along with the increase in Val Loss as the Training Loss dropped, indicated the model's ability to learn to reconstruct valid IDs effectively, while failing to reconstruct invalid ones. Only the validation set had negative IDs at all, so this is expected.
* For the Classifier: The best `ROC-AUC Score was 0.8365`. Looking at `Precision: 0.7532` and `Recall: 0.8389`, clearly the model is better at saying Valid to most IDs, sacrificing a little on correct answers. The model achieved an overall `accuracy of 82.84%`. The current results are a limitation of the current approach, and could present opportunities for future work (see section 7).
* Reconstruction Error & Latent space vector as a Signal: The VAE's reconstruction cross-entropy proved to be the highest-importance feature for the downstream MLP, effectively acting as an anomaly score, signalled by the KL divergence.

## 7. Future Work & Limitations
* Currently, the model operates on a "closed-world" assumption where the input is a pre-extracted ID string. In production environments (e.g., OCR from shipping labels, or unstructured PDF documents), IDs are often embedded within noisy text. Implementing a Sliding Window or Token Segmentation approach may allow the model to extract and validate IDs from unstructured data streams.

* The current VAE architecture utilizes a flattened embedding layer: `x_flat = x_emb.view(batch, -1)`. While computationally efficient, this treats the sequence as a high-dimensional vector and may overlook long-range spatial dependencies between specific character positions (e.g., character at position 2 may relate to character at, say, position 4). Integrating Positional Encodings or transitioning to a Recurrent (LSTM) or Attention-based (Transformer) encoder may help.

## 8. References & Acknowledgments
* Kaggle Dataset: Amazon Product Reviews ([Kaggle/arhamrumi/amazon-product-reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews/data))
* VAE Framework: Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*
* $\beta$-VAE Logic: Higgins, I., et al. (2017). *$\beta$-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*

## 9. Citation & attribution
If you found my work helpful or use any part of this code in your own research or projects, please consider giving the repository a star and citing this work:
```code snippet
@misc{bhati_product_id_classifier_2026,
  author = {Bhati, Divyanshu},
  title = {Product ID Anomaly Detection: A Hybrid VAE-Classifier Approach},
  year = {2026},
  publisher = {GitHub},
  version = {1.0},
  url = {https://github.com/Divyanshu-Bhati/product_id_classifier}
}
```

## 10. License
This project is licensed under the `Apache License 2.0`. You are free to use, modify, and distribute this software/code/logic, provided that you include the original copyright notice and disclaimer. See the [LICENSE](https://github.com/Divyanshu-Bhati/product_id_classifier/blob/main/LICENSE) file for more details.