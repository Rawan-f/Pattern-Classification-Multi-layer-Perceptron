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
wordEmbeddings = load_glove_vectors(glove_file)


# Step 2: Read and process each text file and extract word vectors
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]
    return words


categories = ['animals.txt', 'countries.txt', 'fruits.txt', 'veggies.txt']
word_vectors_by_category = {}

def get_word_embedding(word, embeddings):
    if word in embeddings:
        return embeddings[word]
    else:
        return None


for category in categories:
    file_path = category
    words = read_text_file(file_path)
    word_vectors = {}

    for word in words:
        embedding = get_word_embedding(word, wordEmbeddings)
        if embedding is not None:
            word_vectors[word] = embedding
        #else:       #To print words that do not exist in the loaded GloVe embeddings dictionary.
         #    print(word)

    word_vectors_by_category[category] = word_vectors
    print(word_vectors_by_category[category])