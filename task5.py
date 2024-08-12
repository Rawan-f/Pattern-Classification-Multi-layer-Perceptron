import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, epochs, batch_size):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_weights = None
        self.hidden_layer_biases = None
        self.output_layer_weights = None
        self.output_layer_biases = None
        self.encoder = None

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def train(self, X_train, y_train, X_test, y_test):
        self.encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = self.encoder.fit_transform(np.array(y_train).reshape(-1, 1))
        y_test_encoded = self.encoder.transform(np.array(y_test).reshape(-1, 1))

        self.hidden_layer_weights = np.random.randn(self.input_dim, self.hidden_dim)
        self.hidden_layer_biases = np.zeros((1, self.hidden_dim))
        self.output_layer_weights = np.random.randn(self.hidden_dim, self.output_dim)
        self.output_layer_biases = np.zeros((1, self.output_dim))

        train_accuracies = []
        test_accuracies = []

        for epoch in range(self.epochs):
            loss = 0.0
            all_train_predictions = []
            all_train_ground_truth = []

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train_encoded[i:i + self.batch_size]

                hidden_layer_input = np.dot(X_batch, self.hidden_layer_weights) + self.hidden_layer_biases
                hidden_layer_output = self.relu(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.output_layer_weights) + self.output_layer_biases
                output_layer_output = self.softmax(output_layer_input)

                batch_loss = -np.sum(y_batch * np.log(output_layer_output + 1e-10)) / self.batch_size
                loss += batch_loss

                train_predictions = np.argmax(output_layer_output, axis=1)
                train_ground_truth = np.argmax(y_batch, axis=1)
                all_train_predictions.extend(train_predictions)
                all_train_ground_truth.extend(train_ground_truth)

                d_output = (output_layer_output - y_batch) / self.batch_size
                d_hidden = np.dot(d_output, self.output_layer_weights.T) * (hidden_layer_output > 0)

                self.output_layer_weights -= self.learning_rate * np.dot(hidden_layer_output.T, d_output)
                self.output_layer_biases -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
                self.hidden_layer_weights -= self.learning_rate * np.dot(X_batch.T, d_hidden)
                self.hidden_layer_biases -= self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

            average_loss = loss / (len(X_train) / self.batch_size)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}')

            train_accuracy = np.mean(np.array(all_train_predictions) == np.array(all_train_ground_truth))
            train_accuracies.append(train_accuracy)
            print(f'Training Accuracy : {train_accuracy * 100:.2f}%')

            test_hidden_layer_input = np.dot(X_test, self.hidden_layer_weights) + self.hidden_layer_biases
            test_hidden_layer_output = self.relu(test_hidden_layer_input)
            test_output_layer_input = np.dot(test_hidden_layer_output, self.output_layer_weights) + self.output_layer_biases
            test_output_layer_output = self.softmax(test_output_layer_input)

            test_predictions = np.argmax(test_output_layer_output, axis=1)
            test_ground_truth = np.argmax(y_test_encoded, axis=1)
            test_accuracy = np.mean(test_predictions == test_ground_truth)
            test_accuracies.append(test_accuracy)
            print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        return train_accuracies, test_accuracies

def load_glove_vectors(glove_file):
    word_embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_embeddings[word] = vector
    return word_embeddings

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]
    return words

def get_word_embedding(word, embeddings):
    if word in embeddings:
        return embeddings[word]
    else:
        return None

# Step 1: Load GloVe word embeddings
glove_file = 'glove.6B.50d.txt'  # Replace with your GloVe file path
word_embeddings = load_glove_vectors(glove_file)

# Step 2: Read and process each text file and extract word vectors
categories = ['animals.txt', 'countries.txt', 'fruits.txt', 'veggies.txt']
data = []
labels = []

for category in categories:
    file_path = category
    words = read_text_file(file_path)

    for word in words:
        embedding = get_word_embedding(word, word_embeddings)
        if embedding is not None:
            data.append(embedding)
            labels.append(category)

# Step 3: Split the dataset with class balance using StratifiedShuffleSplit
sl = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sl.split(data, labels):
    X_train, X_test = np.array(data)[train_index], np.array(data)[test_index]
    y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

# Step 4: MLP implementation
    input_dim = len(X_train[0])  # Dimensionality of word embeddings (50 in this case)
    output_dim = 4  # Number of categories (animals, countries, fruits, veggies)
    hidden_layer_dim = 80  # Number of neurons in the hidden layer

    mlp = MLP(input_dim, hidden_layer_dim, output_dim, learning_rate=0.1, epochs=25, batch_size=32)

    train_accuracies, test_accuracies = mlp.train(X_train, y_train, X_test, y_test)

# Step 5:  Plot a figure for train/test classification accuracy for the training epochs
    epochs = range(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy', marker='o')
    plt.plot(epochs, test_accuracies, 'r', label='Test accuracy', marker='o')
    plt.title('Train/Test Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()