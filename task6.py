import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, learning_rate, epochs, batch_size):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_weights = [None] * num_hidden_layers
        self.hidden_layer_biases = [None] * num_hidden_layers
        self.output_layer_weights = None
        self.output_layer_biases = None
        self.encoder = None
        self.hidden_layer_weights[0] = np.random.randn(self.input_dim, hidden_dim)
        self.hidden_layer_biases[0] = np.zeros((1, hidden_dim))
        for i in range(1, num_hidden_layers):
            self.hidden_layer_weights[i] = np.random.randn(hidden_dim, hidden_dim)
            self.hidden_layer_biases[i] = np.zeros((1, hidden_dim))
        self.output_layer_weights = np.random.randn(hidden_dim, self.output_dim)
        self.output_layer_biases = np.zeros((1, self.output_dim))
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

        train_accuracies = []
        test_accuracies = []

        for epoch in range(self.epochs):
            loss = 0.0
            all_train_predictions = []
            all_train_ground_truth = []

            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train_encoded[i:i + self.batch_size]

                hidden_layer_outputs = [X_batch]
                for layer in range(self.num_hidden_layers):
                    hidden_layer_input = np.dot(hidden_layer_outputs[-1], self.hidden_layer_weights[layer]) + self.hidden_layer_biases[layer]
                    hidden_layer_output = self.relu(hidden_layer_input)
                    hidden_layer_outputs.append(hidden_layer_output)

                output_layer_input = np.dot(hidden_layer_outputs[-1], self.output_layer_weights) + self.output_layer_biases
                output_layer_output = self.softmax(output_layer_input)

                batch_loss = -np.sum(y_batch * np.log(output_layer_output + 1e-10)) / self.batch_size
                loss += batch_loss

                train_predictions = np.argmax(output_layer_output, axis=1)
                train_ground_truth = np.argmax(y_batch, axis=1)
                all_train_predictions.extend(train_predictions)
                all_train_ground_truth.extend(train_ground_truth)

                d_output = (output_layer_output - y_batch) / self.batch_size
                d_hidden_layers = [None] * self.num_hidden_layers
                d_hidden_layers[-1] = np.dot(d_output, self.output_layer_weights.T) * (hidden_layer_outputs[-1] > 0)

                for layer in range(self.num_hidden_layers - 2, -1, -1):
                    d_hidden_input = np.dot(d_hidden_layers[layer + 1], self.hidden_layer_weights[layer + 1].T)
                    d_hidden_layers[layer] = d_hidden_input * (hidden_layer_outputs[layer + 1] > 0)

                for layer in range(self.num_hidden_layers):
                    self.hidden_layer_weights[layer] -= self.learning_rate * np.dot(hidden_layer_outputs[layer].T, d_hidden_layers[layer])
                    self.hidden_layer_biases[layer] -= self.learning_rate * np.sum(d_hidden_layers[layer], axis=0, keepdims=True)

                self.output_layer_weights -= self.learning_rate * np.dot(hidden_layer_outputs[-1].T, d_output)
                self.output_layer_biases -= self.learning_rate * np.sum(d_output, axis=0, keepdims=True)

            average_loss = loss / (len(X_train) / self.batch_size)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}')

            train_accuracy = np.mean(np.array(all_train_predictions) == np.array(all_train_ground_truth))
            train_accuracies.append(train_accuracy)
            print(f'Training Accuracy : {train_accuracy * 100:.2f}%')

            test_hidden_layer_outputs = [X_test]
            for layer in range(self.num_hidden_layers):
                test_hidden_layer_input = np.dot(test_hidden_layer_outputs[-1], self.hidden_layer_weights[layer]) + self.hidden_layer_biases[layer]
                test_hidden_layer_output = self.relu(test_hidden_layer_input)
                test_hidden_layer_outputs.append(test_hidden_layer_output)

            test_output_layer_input = np.dot(test_hidden_layer_outputs[-1], self.output_layer_weights) + self.output_layer_biases
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

    # Vary the number of hidden layers and evaluate the performance
    input_dim = len(X_train[0])  # Dimensionality of word embeddings (50 in this case)
    output_dim = 4  # Number of categories (animals, countries, fruits, veggies)

    # Define different numbers of hidden layers (k) to evaluate
    hidden_layer_values = [1, 2, 3]

    train_accuracies_list = []
    test_accuracies_list = []

    for num_hidden_layers in hidden_layer_values:
        hidden_layer_dim = 80  # Number of neurons in the hidden layer
        mlp = MLP(input_dim, hidden_layer_dim, output_dim, num_hidden_layers, learning_rate=0.1, epochs=25, batch_size=32)
        train_accuracies, test_accuracies = mlp.train(X_train, y_train, X_test, y_test)
        train_accuracies_list.append(train_accuracies[-1])
        test_accuracies_list.append(test_accuracies[-1])

    # Plot the results
    plt.figure()
    plt.plot(hidden_layer_values, train_accuracies_list, marker='o', label='Training Accuracy')
    plt.plot(hidden_layer_values, test_accuracies_list, marker='o', label='Testing Accuracy')
    plt.xlabel('Number of Hidden Layers (k)')
    plt.ylabel('Accuracy')
    plt.title('MLP Classifier Performance vs. Number of Hidden Layers')
    plt.legend()
    plt.grid(True)
    plt.show()
