import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm


class TextDataset(Dataset):
    """
    Dataset class for handling text data and preparing it for token classification.
    """
    def __init__(self, texts, tokenizer, max_length=128):
        """
        Initializes the dataset with texts, tokenizer, and maximum token length.

        Parameters:
        - texts: List of strings to be tokenized.
        - tokenizer: Hugging Face tokenizer to process the text.
        - max_length: Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the number of text samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves and tokenizes a single text sample by index.

        Parameters:
        - idx: Index of the text sample.

        Returns:
        - A dictionary containing tokenized input IDs and attention masks.
        """
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


class Preprocessing:
    """
    A preprocessing pipeline for replacing named entities in text data with generic placeholders.
    """
    def __init__(self, data, model_tokenizer_path="ner_model", batch_size=32):
        """
        Initializes the preprocessing object with model path, data, and batch size.

        Parameters:
        - data: List of text samples to preprocess.
        - model_tokenizer_path: Path to the Hugging Face model for token classification.
        - batch_size: Batch size for data processing during inference.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_tokenizer_path = model_tokenizer_path
        self.data = data
        self.batch_size = batch_size

    def __get_model__(self):
        """
        Loads the pre-trained token classification model and tokenizer.

        Returns:
        - model: Hugging Face model for token classification.
        - tokenizer: Hugging Face tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_tokenizer_path).to(self.device)
        return model, tokenizer

    def __replace_names__(self, data, names_all, target):
        """
        Replaces all identified names in the text data with a placeholder.

        Parameters:
        - data: Original list of text samples.
        - names_all: List of names identified for each text sample.
        - target: Placeholder string to replace the names.

        Returns:
        - List of updated text samples with names replaced by the placeholder.
        """
        new_data = []
        for text, names in tqdm(zip(data, names_all), "Replacing Names"):
            new_text = text
            for name in names:
                new_text = new_text.replace(name, target)
            new_data.append(new_text)
        return new_data

    def __is_subtoken__(self, token):
        """
        Determines if a token is a subtoken (continuation of a previous token).

        Parameters:
        - token: A single token string.

        Returns:
        - True if the token is a subtoken; False otherwise.
        """
        if token:
            return not ("▁" == token[0])  # Subtokens usually start without a special character.
        return False

    def apply(self):
        """
        Applies the preprocessing pipeline: detects and replaces named entities.

        Returns:
        - List of text samples with named entities replaced by a placeholder.
        """
        model, tokenizer = self.__get_model__()  # Load model and tokenizer
        id2label = model.config.id2label  # Mapping of label IDs to their string labels

        # Create dataset and dataloader for batching
        dataset = TextDataset(self.data, tokenizer, max_length=128)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()  # Set the model to evaluation mode
        names_all = []  # To store all detected names
        with torch.no_grad():  # Disable gradient calculations
            for batch in tqdm(dataloader, "Preprocessing"):
                input_ids_batch = batch["input_ids"].to(self.device)
                attention_mask_batch = batch["attention_mask"].to(self.device)
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits  # Output logits from the model

                predictions_batch = torch.argmax(logits, dim=-1).cpu().numpy()  # Get predicted label indices

                # Process predictions for each batch
                for predictions, input_ids in zip(predictions_batch, input_ids_batch.cpu().numpy()):
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Convert IDs to tokens
                    names = []
                    predictions_label = [id2label[p] for p in predictions]  # Map label IDs to labels
                    
                    entity_started = False  # Flag to track entity boundaries
                    entity = []  # Store token IDs of the current entity
                    for input_id, pred, token in zip(input_ids, predictions_label, tokens):
                        if pred == "I-PER" and not self.__is_subtoken__(token):
                            # Start a new entity if necessary
                            if entity:
                                names.append(tokenizer.decode(entity, skip_special_tokens=True))
                                entity = []
                                entity_started = False
                            entity.append(input_id)
                            entity_started = True
                            continue
                        if entity_started:
                            # Continue the entity or close it based on predictions
                            if pred == "I-PER" or self.__is_subtoken__(token):
                                entity.append(input_id)
                            else:
                                names.append(tokenizer.decode(entity, skip_special_tokens=True))
                                entity = []
                                entity_started = False

                    if entity:
                        # Add the last entity if any
                        names.append(tokenizer.decode(entity, skip_special_tokens=True))
                        entity = []
                        entity_started = False

                    names_all.append(names)  # Append detected names for the text

        # Replace identified names with the placeholder
        return self.__replace_names__(self.data, names_all, target="Person")
