import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Step 1: Load GloVe word embeddings
def load_glove_vectors(glove_file):
    word_embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_embeddings[word] = vector
    return word_embeddings

glove_file = 'glove.6B.50d.txt'  # Replace with your GloVe file path
word_embeddings = load_glove_vectors(glove_file)

# Step 2: Read and process each text file and extract word vectors
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]
    return words

categories = ['animals.txt', 'countries.txt', 'fruits.txt', 'veggies.txt']
data = []
labels = []

def get_word_embedding(word, embeddings):
    if word in embeddings:
        return embeddings[word]
    else:
        return None

for category in categories:
    file_path = category
    words = read_text_file(file_path)

    for word in words:
        embedding = get_word_embedding(word, word_embeddings)
        if embedding is not None:
            data.append(embedding)
            labels.append(category[:-4])

# Step 3: Split the dataset with class balance using StratifiedShuffleSplit
sl = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sl.split(data, labels):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]