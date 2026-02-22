import torch
import random

def decode_sequence(encoded_seq, idx2char):
    """Decode a sequence of token indices into a string.
    
    Stops decoding when encountering the <PAD> token (assumed index 0).
    """
    decoded = []
    for token in encoded_seq:
        if token == 0:  # <PAD> token
            break
        decoded.append(idx2char.get(token, '<UNK>'))
    return ''.join(decoded)

def show_random_reconstructions(model, X_test_padded, char2idx, idx2char, num_samples=10):
    """Randomly pick 'num_samples' examples from the test set and display:
       - the original sequence,
       - its reconstruction,
       - and the latent representation (encoder mean).
    """
    indices = random.sample(range(len(X_test_padded)), num_samples)
    model.eval()
    print("\nRandom Reconstructions:")
    for i in indices:
        x_sample = torch.tensor(X_test_padded[i], dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            recon_logits, z_mean, z_log_var = model(x_sample)
            # Get predicted token IDs along vocab dimension
            pred_token_ids = torch.argmax(recon_logits, dim=-1).squeeze(0).cpu().numpy().tolist()
            original_tokens = x_sample.squeeze(0).cpu().numpy().tolist()
            
            original_str = decode_sequence(original_tokens, idx2char)
            recon_str = decode_sequence(pred_token_ids, idx2char)
            print("Original:       ", original_str)
            print("Reconstruction: ", recon_str)
            # print("Latent vector (mean):", z_mean.squeeze(0).cpu().numpy())
            print("-" * 50)

def main(model, X_test_padded, char2idx, idx2char):
    """Main function to visualize reconstructions and latent feature effects.
    
    Call this function from your inference script.
    """
    show_random_reconstructions(model, X_test_padded, char2idx, idx2char)
    
if __name__ == "__main__":
    print("This is the visualizer module. Call the main() function from inference script.")