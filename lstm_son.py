import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

from keras.layers import Dense, LSTM, Flatten, Embedding
from keras.utils import to_categorical
from keras.backend import clear_session
import pickle

# Load pre-trained word vectors
filename = "C:/Users/Pc/Desktop/proje-1/GoogleNews-vectors-negative300.bin"

start = time.time()
google_embeddings = KeyedVectors.load_word2vec_format(filename, binary=True)

print("Load time (seconds): ", (time.time() - start))
glove_file = "C:/Users/Pc/Desktop/proje-1/glove.6B/glove.6B.300d.txt"
glove_word2vec_file = "glove.6B.300d.txt.word2vec"

glove2word2vec(glove_file, glove_word2vec_file)
start = time.time()

glove_embeddings = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)

print("Load time (seconds): ", (time.time() - start))

data = pd.read_csv("C:/Users/Pc/Desktop/proje-1/preprocessed.csv")

data.head()
data['X-Folder'].unique()

def label_encoder(data):
    class_le = LabelEncoder()
    y = class_le.fit_transform(data['X-Folder'])
    return y

y = label_encoder(data)
input_data = data['text']
X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.1)
Y_train = to_categorical(y_train, 20)
Y_test = to_categorical(y_test, 20)
t = Tokenizer()

# fit the tokenizer on the docs
t.fit_on_texts(input_data)
vocab_size = len(t.word_index) + 1

# integer encode the documents
X_train_encoded_docs = t.texts_to_sequences(X_train)
X_test_encoded_docs = t.texts_to_sequences(X_test)
max_length = 150
X_train_padded_docs = pad_sequences(X_train_encoded_docs, maxlen=max_length, padding='post')
X_test_padded_docs = pad_sequences(X_test_encoded_docs, maxlen=max_length, padding='post')

print(X_train_padded_docs[0])

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in t.word_index.items():
    try:
        embedding_vector = google_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass

# define the model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(LSTM(100, dropout=0.2))
model.add(Flatten())
model.add(Dense(20, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
model.summary()
clear_session()

history = model.fit(X_train_padded_docs, Y_train, epochs=60, verbose=1, validation_split=0.1)
accr = model.evaluate(X_test_padded_docs, Y_test)
print("Test Set: \n Loss: {:0.3f}\n Accuracy: {:0.3f}".format(accr[0], accr[1]))
plt.title("Word2Vec Loss")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()
plt.title("Word2Vec Accuracy")
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.show()

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in t.word_index.items():
    try:
        embedding_vector = glove_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass

# define the model
model2 = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
model2.add(e)
model2.add(LSTM(100, dropout=0.2))
model2.add(Flatten())
model2.add(Dense(20, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
model2.summary()

# fit the model
history2 = model2.fit(X_train_padded_docs, Y_train, epochs=60, verbose=1, validation_split=0.1)
accr2 = model2.evaluate(X_test_padded_docs, Y_test)
print("Test Set: \n Loss: {:0.3f}\n Accuracy: {:0.3f}".format(accr2[0], accr2[1]))

# plot the loss
plt.title("Glove Word2Vec Loss")
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='validation')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.show()
plt.title("GloVe Word2Vec Accuracy")
plt.plot(history2.history['accuracy'], label='train')
plt.plot(history2.history['val_accuracy'], label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.show()

# define a new model for training data embeddings
model3 = Sequential()
model3.add(Embedding(vocab_size, 300, input_length=max_length))
model3.add(LSTM(100, dropout=0.4))
model3.add(Flatten())
model3.add(Dense(20, activation='softmax'))

# compile the model
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
model3.summary()
history3 = model3.fit(X_train_padded_docs, Y_train, validation_split=0.1, epochs=20, verbose=1)

# save the trained embeddings
own_embeddings = model3.get_layer('embedding_1').get_weights()[0]

custom_w2v = {}
for word, index in t.word_index.items():
    custom_w2v[word] = own_embeddings[index]

# save to file
with open("C:/Users/Pc/Desktop/proje-1/own_embeddings.pkl", "wb") as handle:
    pickle.dump(custom_w2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

# evaluate the model
accr3 = model3.evaluate(X_test_padded_docs, Y_test)
print("Test Set:\n  Loss: {:0.3f}\n   Accuracy: {:0.3f}".format(accr3[0], accr3[1]))

# plot the loss
plt.title("Loss")
plt.plot(history3.history['loss'], label='train')
plt.plot(history3.history['val_loss'], label='validation')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.show()

# plot the accuracy
plt.title("Accuracy")
plt.plot(history3.history['accuracy'], label='train')
plt.plot(history3.history['val_accuracy'], label='validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.show()

# summarize results
res_data = {
    "Technique": ['Word2Vec', 'GloVe', 'Training data Embeddings'],
    "test accuracy": [accr[1], accr2[1], accr3[1]]
}
result = pd.DataFrame(res_data)
print(result)
# Google News Word2Vec modeli
model.save("C:/Users/Pc/Desktop/proje-1/google_word2vec_model.h5")

# GloVe Word2Vec modeli
model2.save("C:/Users/Pc/Desktop/proje-1/glove_word2vec_model.h5")

# Eğitim verileriyle eğitilmiş model
model3.save("C:/Users/Pc/Desktop/proje-1/custom_embeddings_model.h5")

