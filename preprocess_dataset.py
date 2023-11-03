import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_dataset(dataset_path, max_sequence_length):
    sequences = []
    labels = []
    classes = os.listdir(dataset_path)
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            sequence = str(SeqIO.read(file_path, "fasta").seq)
            sequences.append(sequence)
            labels.append(class_name)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    label_mapping = {class_name: i for i, class_name in enumerate(set(labels))}
    labels = [label_mapping[label] for label in labels]
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)
    
    return train_sequences, test_sequences, train_labels, test_labels, label_mapping
