import random
import nltk
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Define a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, max_length=10):
        # Load the file and split it into lines
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        
        # Tokenizer
        self.tokenizer = get_tokenizer('basic_english')
        
        # Maximum length of tokens (fixed length)
        self.max_length = max_length

    def __len__(self):
        # Returns the total number of lines in the file
        return len(self.lines)

    def __getitem__(self, idx):
        # Tokenize the line at the given index
        text = self.lines[idx].strip()  # Remove any trailing newlines or spaces
        tokens = self.tokenizer(text)
        
        # Convert tokens to a tensor and pad/truncate them to max_length
        token_ids = torch.tensor([ord(token[0]) for token in tokens], dtype=torch.long)  # Simple token encoding for example
        
        # Pad or truncate the token_ids to the fixed max_length
        if len(token_ids) < self.max_length:
            token_ids = torch.cat([token_ids, torch.zeros(self.max_length - len(token_ids), dtype=torch.long)])
        else:
            token_ids = token_ids[:self.max_length]
        
        # Return tokenized text and a dummy label (e.g., 0)
        label = 0  # Assign dummy label (for example purposes)
        return token_ids, label

class PreProcessText(TextDataset):
    def lowercase_text(self,output_data):
        """
        Converts all characters in the input sentence to lowercase.
        """
        idx = 0
        for lines in self.lines:
            lines = lines.lower()
            self.lines[idx] = lines
            idx = idx + 1
            output_data.append(str(lines))

    def remove_stop_words(self,output_data):
        """
        Removes common stop words from the input sentence.
        """
        stop_words = {'the', 'is', 'and', 'in', 'on', 'at', 'to', 'with', 'a', 'an'}  # Example set of stop words
        idx = 0
        for lines in self.lines:
            words = lines.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            lines = ' '.join(filtered_words)
            self.lines[idx] = lines
            idx = idx + 1
            output_data.append(str(lines))
            output_data.append("\n")

class AugmentText(TextDataset):
    def synonym_replacement(self,output_data,n=1):
        idx = 0
        for lines in self.lines:
            words = word_tokenize(lines)
            new_words = words.copy()
            random.shuffle(words)
            
            num_replaced = 0
            for word in words:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    synonym = random.choice(synonyms)
                    new_words = [synonym if w == word else w for w in new_words]
                    num_replaced += 1
                if num_replaced >= n:
                    break
            
            lines = ' '.join(new_words)
            self.lines[idx] = lines
            idx = idx + 1
            output_data.append(str(lines))
            output_data.append("\n")

    def get_synonyms(self,word):
        """
        Retrieves synonyms for a given word using WordNet.
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def random_insertion(self,output_data, n=1):
        """
        Randomly inserts 'n' synonyms of random words into the sentence.
        """
        idx = 0
        for lines in self.lines:
            words = word_tokenize(lines)
            for _ in range(n):
                new_word = self.get_random_synonym(words)
                if new_word:
                    insert_position = random.randint(0, len(words))
                    words.insert(insert_position, new_word)
            
            lines = ' '.join(words)
            self.lines[idx] = lines
            idx = idx + 1
            output_data.append(str(lines))
            output_data.append("\n")

    def get_random_synonym(self,words):
        """
        Gets a random synonym of a random word from the list of words.
        """
        random_word = random.choice(words)
        synonyms = self.get_synonyms(random_word)
        if synonyms:
            return random.choice(synonyms)
        return None