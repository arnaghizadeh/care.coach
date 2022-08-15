import sys

import numpy as np
import torch
from transformers import GPT2Tokenizer

from FinalSubmission.fileIO import get_file

pretrained_hf_model = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(pretrained_hf_model, pad_token='<|endoftext|>', )
labels = {'sentimental': 0, 'afraid': 1, 'proud': 2, 'faithful': 3, 'terrified': 4, 'joyful': 5,
          'angry': 6, 'sad': 7, 'jealous': 8, 'grateful': 9, 'prepared': 10, 'embarrassed': 11, 'excited': 12,
          'annoyed': 13, 'lonely': 14, 'ashamed': 15, 'guilty': 16, 'surprised': 17, 'nostalgic': 18,
          'confident': 19, 'furious': 20, 'disappointed': 21, 'caring': 22, 'trusting': 23, 'disgusted': 24,
          'anticipating': 25, 'anxious': 26, 'hopeful': 27, 'content': 28, 'impressed': 29, 'apprehensive': 30,
          'devastated': 31}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, prompt_texts, response_texts, lbls):
        #we dont need labels for now, but I added this for the future
        self.labels = [labels[label] for label in lbls]
        self.texts = []
        #create a list with A, B format
        for idx in range(len(prompt_texts)):
            str = "A: " + prompt_texts[idx] + "\n" + "B: " + response_texts[idx] + " <EMOTION_TYPE> " + lbls[
                idx] + " <|endoftext|>\n"
            self.texts.append(str)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
