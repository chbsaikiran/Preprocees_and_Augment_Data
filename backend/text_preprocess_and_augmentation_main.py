import text_preprocess_and_augmentation_classes
import nltk
from text_preprocess_and_augmentation_classes import TextDataset,PreProcessText,AugmentText
from torch.utils.data import DataLoader

def show_original_data(file_path):
    output_data = []
    # Create Dataset instance with a fixed maximum sequence length (for padding)
    train_dataset = TextDataset(file_path, max_length=10)

    # Collect lines data
    output_data.append("Original Lines:")
    for lines in train_dataset.lines:
        output_data.append(str(lines))

    # Create DataLoader instance
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # Collect batch data
    for batch_idx, (token_ids, labels) in enumerate(train_loader):
        output_data.append(f"Batch {batch_idx + 1}")
        output_data.append(f"Token IDs: {token_ids}")
        output_data.append(f"Labels: {labels}")
        output_data.append("-------")

    return "\n".join(output_data)

def show_lowercase_data(file_path):
    output_data = []
    # Create Dataset instance
    train_dataset_lower = PreProcessText(file_path, max_length=10)
    output_data.append("\n")
    train_dataset_lower.lowercase_text(output_data)
    output_data.append("\n")
    output_data.append("\n")

    # Create DataLoader instance
    train_loader_lower = DataLoader(train_dataset_lower, batch_size=2, shuffle=True)
    
    # Collect batch data
    for batch_idx, (token_ids, labels) in enumerate(train_loader_lower):
        output_data.append(f"Batch {batch_idx + 1}")
        output_data.append(f"Token IDs: {token_ids}")
        output_data.append(f"Labels: {labels}")
        output_data.append("-------")
    output_data.append("\n")
    output_data.append("\n")
    

    return "\n".join(output_data)

def show_after_remove_stop_words_data(file_path):
    output_data = []
    # Create Dataset instance
    train_dataset_remove_stop_words = PreProcessText(file_path, max_length=10)
    output_data.append("\n")
    train_dataset_remove_stop_words.remove_stop_words(output_data)
    output_data.append("\n")
    output_data.append("\n")

    # Create DataLoader instance
    train_loader_remove_stop_words = DataLoader(train_dataset_remove_stop_words, batch_size=2, shuffle=True)
    
    # Collect batch data
    for batch_idx, (token_ids, labels) in enumerate(train_loader_remove_stop_words):
        output_data.append(f"Batch {batch_idx + 1}")
        output_data.append(f"Token IDs: {token_ids}")
        output_data.append(f"Labels: {labels}")
        output_data.append("-------")
    output_data.append("\n")
    output_data.append("\n")

    return "\n".join(output_data)

def show_after_synonym_replacement_data(file_path):
    output_data = []
    # Create Dataset instance
    train_dataset_synonym_replacement = AugmentText(file_path, max_length=10)
    output_data.append("\n")
    train_dataset_synonym_replacement.synonym_replacement(output_data)
    output_data.append("\n")
    output_data.append("\n")

    # Create DataLoader instance
    train_loader_synonym_replacement = DataLoader(train_dataset_synonym_replacement, batch_size=2, shuffle=True)
    
    # Collect batch data
    for batch_idx, (token_ids, labels) in enumerate(train_loader_synonym_replacement):
        output_data.append(f"Batch {batch_idx + 1}")
        output_data.append(f"Token IDs: {token_ids}")
        output_data.append(f"Labels: {labels}")
        output_data.append("-------")
    output_data.append("\n")
    output_data.append("\n")

    return "\n".join(output_data)

def show_after_random_insertion_data(file_path):
    output_data = []
    # Create Dataset instance
    train_dataset_random_insertion = AugmentText(file_path, max_length=10)
    output_data.append("\n")
    train_dataset_random_insertion.random_insertion(output_data)
    output_data.append("\n")
    output_data.append("\n")

    # Create DataLoader instance
    train_loader_random_insertion = DataLoader(train_dataset_random_insertion, batch_size=2, shuffle=True)
    
    # Collect batch data
    for batch_idx, (token_ids, labels) in enumerate(train_loader_random_insertion):
        output_data.append(f"Batch {batch_idx + 1}")
        output_data.append(f"Token IDs: {token_ids}")
        output_data.append(f"Labels: {labels}")
        output_data.append("-------")
    output_data.append("\n")
    output_data.append("\n")

    return "\n".join(output_data)