import os
import json
import numpy as np
import pandas as pd

class DataCreator:
    def __init__(self, input_path, training_history, seed, experiment_mode=False):
        self.input_path = input_path
        self.training_history = training_history
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.vae_data = "vae_set.csv"
        self.cls_data = "cls_set.csv"
        self.test_data = "test_set.csv"
        self.wrong_chars_data = "wrong_chars.csv"
        
    def _filters(self, data_dict, data_filters):
        """
        Cleans all datasets in the dictionary based on config parameters.
        Ensures X and y alignment remains intact.
        """
        max_len = data_filters.get("max_len", 14)
        min_len = data_filters.get("min_len", 10)
        remove_non_ascii = data_filters.get("remove_non_ascii", True)
        
        illegal_set = set(pd.read_csv(os.path.join(self.input_path, self.wrong_chars_data))['found_chars'].tolist())

        def is_valid(s):
            s = str(s)
            if not (min_len <= len(s) <= max_len):
                return False
            if remove_non_ascii:
                if any(char in illegal_set for char in s):
                    return False
                if any(ord(char) > 127 for char in s):
                    return False
            return True

        filtered_data = {}
        for key in list(data_dict.keys()):
            if key.startswith("X_"):
                suffix = key.split("_")[1]
                y_key = f"y_{suffix}"
                X_list = data_dict[key]
                if y_key in data_dict:
                    y_list = data_dict[y_key]
                    clean_pairs = [(x, y) for x, y in zip(X_list, y_list) if is_valid(x)]
                    if clean_pairs:
                        filtered_data[key], filtered_data[y_key] = map(list, zip(*clean_pairs))
                    else:
                        filtered_data[key], filtered_data[y_key] = [], []
                else:
                    filtered_data[key] = [x for x in X_list if is_valid(x)]
        return filtered_data

    def load_data(self, data_filters, task_type):
        def encode_labels(series):
            return series.map({'yes': 1, 'no': 0}).tolist()

        data = {}
        if task_type == "train_vae":
            df = pd.read_csv(os.path.join(self.input_path, self.vae_data))
            if self.experiment_mode:
                df = df.sample(n=10_000, random_state=self.seed).reset_index(drop=True)
                print("In experiment mode: Using 10_000 samples only.")
            train_df = df[df["split"] == "train"]
            eval_df = df[df["split"] == "eval"]
            data["X_train"] = train_df["input"].tolist()
            data["X_eval"]  = eval_df["input"].tolist()
            data["y_eval"]  = encode_labels(eval_df["is_pd_id"])

        elif task_type == "train_cls":
            df = pd.read_csv(os.path.join(self.input_path, self.cls_data))
            if self.experiment_mode:
                df = df.sample(n=5_000, random_state=self.seed).reset_index(drop=True)
                print("In experiment mode: Using 5_000 samples only.")
            train_df = df[df["split"] == "train"]
            eval_df = df[df["split"] == "eval"]
            data["X_train"] = train_df["input"].tolist()
            data["y_train"] = encode_labels(train_df["is_pd_id"])
            data["X_eval"]  = eval_df["input"].tolist()
            data["y_eval"]  = encode_labels(eval_df["is_pd_id"])

        elif task_type == "test":
            df = pd.read_csv(os.path.join(self.input_path, self.test_data))
            if self.experiment_mode:
                df = df.sample(n=5_000, random_state=self.seed).reset_index(drop=True)
                print("In experiment mode: Using 5_000 samples only.")
            test_df = df[df["split"] == "test"]
            data["X_test"] = test_df["input"].tolist()
            data["y_test"] = encode_labels(test_df["is_pd_id"])

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        for key, value in data.items():
            print(f"Loaded {len(value)} items for {key}")
            
        return self._filters(data, data_filters)

    def vae_feature_engineer(self, X_train, X_val, max_length):
        def build_char_vocabulary(text_list):
            unique_chars = set()
            for code in text_list:
                for ch in str(code):
                    unique_chars.add(ch)
            
            unique_chars = sorted(list(unique_chars))
            char2idx = {'<PAD>': 0, '<UNK>': 1} # <PAD> token is set sto 0 to ensure its gradient is ignored during training
            for i, ch in enumerate(unique_chars, start=2):
                char2idx[ch] = i
            idx2char = {v: k for k, v in char2idx.items()}
            return char2idx, idx2char

        char2idx, idx2char = build_char_vocabulary(X_train)
        unk_id = char2idx['<UNK>']

        def process_set(text_list):
            padded = np.zeros((len(text_list), max_length), dtype=np.int64)
            for i, code in enumerate(text_list):
                encoded = [char2idx.get(ch, unk_id) for ch in str(code)]
                length = min(len(encoded), max_length)
                padded[i, :length] = encoded[:length]
            return padded

        X_train_padded = process_set(X_train)
        X_val_padded = process_set(X_val)
        return X_train_padded, X_val_padded, char2idx, idx2char

    def run_vae(self, data_filters, task_type):
        data = self.load_data(data_filters=data_filters, task_type=task_type)
        X_train, X_val, y_val = data["X_train"], data["X_eval"], data["y_eval"]
        max_length = max(len(x) for x in X_train)
        print("Max seq len:", max_length, "found for MPN:", max(X_train, key=len))
        X_train_padded, X_val_padded, char2idx, idx2char = self.vae_feature_engineer(X_train, X_val, max_length)
        # Save vocabulary (with training max_length) for inference consistency.
        vocab_path = os.path.join(self.training_history, "vocab.json")
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w") as f:
            json.dump({"char2idx": char2idx, "idx2char": idx2char, "max_length": max_length}, f)
        return max_length, X_train_padded, X_val_padded, y_val, char2idx, idx2char
    
    def run_cls(self, data_filters, task_type):
        data = self.load_data(data_filters=data_filters, task_type=task_type)
        X_train, X_val, y_train, y_val = data["X_train"], data["X_eval"], data["y_train"], data["y_eval"]
        return X_train, X_val, y_train, y_val
    
    def run_test(self, data_filters, task_type):
        data = self.load_data(data_filters=data_filters, task_type=task_type)
        X_test, y_test = data["X_test"], data["y_test"]
        return X_test, y_test
    
    def load_training_vocab(self):
        # Loads the fixed vocabulary (and training max_length) from the saved JSON.
        vocab_path = os.path.join(self.training_history, "vocab.json")
        if not os.path.exists(vocab_path):
            raise ValueError("Vocabulary file not found. Please retrain the model to generate a vocabulary.")
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        char2idx = vocab["char2idx"]
        idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
        train_max_length = vocab.get("max_length", None)
        if train_max_length is None:
            raise ValueError("Training max length not found in vocabulary file. Please retrain the model.")
        return train_max_length, char2idx, idx2char

    def pad_truncate_data(self, data_list, fixed_max_length, char2idx):
        # Encodes and pads (or truncates) the provided list to the fixed training max_length.
        def encode_sequence(code, char2idx):
            unk_id = char2idx.get("<UNK>", 1)
            return [char2idx.get(ch, unk_id) for ch in code]
        all_encoded = [encode_sequence(x, char2idx) for x in data_list]
        padded = np.zeros((len(all_encoded), fixed_max_length), dtype=np.int64)
        for i, seq in enumerate(all_encoded):
            truncated = seq[:fixed_max_length]
            padded[i, :len(truncated)] = truncated
        return padded
    
    def decode_sequence(self, encoded_seq, idx2char):
        """Decodes a list of token indices into a string, stopping at the <PAD> token (index 0)."""
        decoded = []
        for token in encoded_seq:
            if token == 0:
                break
            decoded.append(idx2char.get(token, '<UNK>'))
        return ''.join(decoded)