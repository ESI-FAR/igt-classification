import torch
from torch.utils.data import Dataset


class IGTClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = self._create_label_map()
        
    def _create_label_map(self):
        unique_labels = set()
        for label_list in self.labels:
            unique_labels.update(label_list)
        return {label: i for i, label in enumerate(sorted(unique_labels))}
    
    def get_label_map(self):
        return self.label_map
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # 
        # (BERT models chunk into subwords, we need to map class labels for tokens to these subword chunks)
        #

        tokens = self.texts[idx]
        labels = self.labels[idx]

        # Ensure tokens and labels have the same length
        if len(tokens) != len(labels):
            print(f"Warning: Tokens and labels have different lengths at index {idx}")
            print(f"Tokens: {len(tokens)}, Labels: {len(labels)}")
            # Adjust length to avoid mismatches
            min_len = min(len(tokens), len(labels))
            tokens, labels = tokens[:min_len], labels[:min_len]

        # Tokenize and get word IDs
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokenized_inputs["input_ids"].squeeze(0)
        attention_mask = tokenized_inputs["attention_mask"].squeeze(0)
        word_ids = tokenized_inputs.word_ids()  # Map tokens back to original words

        # Initialize aligned labels
        aligned_labels = [-100] * len(word_ids)

        previous_word_id = None
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # Special tokens get ignored
                aligned_labels[i] = -100
            elif word_id != previous_word_id:
                # Assign label to first subword
                if word_id < len(labels):  # Prevent index error
                    aligned_labels[i] = self.label_map.get(labels[word_id], -100)
                else:
                    print(f"Warning: word_id {word_id} out of range for labels with length {len(labels)}")
                    aligned_labels[i] = -100  # Ignore out-of-range labels
            else:
                # Assign "I-" if previous word was "B-"
                if previous_word_id is not None and previous_word_id < len(labels):
                    previous_label = labels[previous_word_id]
                    if previous_label.startswith("B-"):
                        i_label = "I-" + previous_label[2:]
                        aligned_labels[i] = self.label_map.get(i_label, self.label_map.get(previous_label, -100))
                    else:
                        aligned_labels[i] = self.label_map.get(previous_label, -100)
                else:
                    aligned_labels[i] = -100  # Ignore unknown cases

            previous_word_id = word_id

        # Convert labels to tensor
        aligned_labels = torch.tensor(aligned_labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels
        }
