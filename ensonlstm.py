import numpy as np
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
from keras.layers import Dense, LSTM, Flatten, Embedding, Dropout
from keras.utils import to_categorical
from keras.backend import clear_session
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.callbacks import EarlyStopping

# Load pre-trained word vectors
filename = "C:/Users/Pc/Desktop/proje-1/GoogleNews-vectors-negative300.bin"
google_embeddings = KeyedVectors.load_word2vec_format(filename, binary=True)

glove_file = "C:/Users/Pc/Desktop/proje-1/glove.6B/glove.6B.300d.txt"
glove_word2vec_file = "glove.6B.300d.txt.word2vec"
glove2word2vec(glove_file, glove_word2vec_file)

glove_embeddings = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)

data = pd.read_csv("C:/Users/Pc/Desktop/proje-1/spam_ham_dataset.csv")
data = data.drop(['Unnamed: 0', 'label'], axis=1)
data = data.rename(columns={"label_num": "Label"})
data.info()

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9]:\S+|subject:\S+|nbsp"

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

data.text = data.text.apply(lambda x: preprocess(x))
data.head()

x = data['text']
y = data['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
print("Train Data size:", len(x_train))
print("Test Data size", len(x_test))

def label_encoder(data):
    class_le = LabelEncoder()
    y = class_le.fit_transform(data['Label'])
    return y

y = label_encoder(data)
input_data = data['text']
t = Tokenizer()

# fit the tokenizer on the docs
t.fit_on_texts(input_data)
vocab_size = len(t.word_index) + 1

# integer encode the documents
x_train_encoded_docs = t.texts_to_sequences(x_train)
x_test_encoded_docs = t.texts_to_sequences(x_test)
max_length = 150
x_train_padded_docs = pad_sequences(x_train_encoded_docs, maxlen=max_length, padding='post')
x_test_padded_docs = pad_sequences(x_test_encoded_docs, maxlen=max_length, padding='post')

print(x_train_padded_docs[0])

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in t.word_index.items():
    try:
        embedding_vector = google_embeddings[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# define the model
modeld = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
modeld.add(e)
modeld.add(LSTM(100, dropout=0.2, return_sequences=True))
modeld.add(LSTM(100, dropout=0.2))
modeld.add(Dense(2, activation='softmax'))

# compile the model
modeld.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
modeld.summary()
clear_session()

history = modeld.fit(x_train_padded_docs, y_train, epochs=10, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
accr = modeld.evaluate(x_test_padded_docs, y_test)
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
model2d = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
model2d.add(e)
model2d.add(LSTM(100, dropout=0.2, return_sequences=True))
model2d.add(LSTM(100, dropout=0.2))
model2d.add(Dense(2, activation='softmax'))

# compile the model
model2d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
model2d.summary()

# fit the model
history2 = model2d.fit(x_train_padded_docs, y_train, epochs=10, verbose=1, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
accr2 = model2d.evaluate(x_test_padded_docs, y_test)
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
model3d = Sequential()
model3d.add(Embedding(vocab_size, 300, input_length=max_length))
model3d.add(LSTM(100, dropout=0.4, return_sequences=True))
model3d.add(LSTM(100, dropout=0.4))
model3d.add(Dense(2, activation='softmax'))

# compile the model
model3d.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# summarize the model
model3d.summary()
history3 = model3d.fit(x_train_padded_docs, y_train, validation_split=0.2, epochs=20, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# save the trained embeddings
own_embeddingsd = model3d.get_layer('embedding_1').get_weights()[0]

custom_w2v = {}
for word, index in t.word_index.items():
    custom_w2v[word] = own_embeddingsd[index]

# save to file
with open("C:/Users/Pc/Desktop/proje-1/deneme_kendi.pkl", "wb") as handle:
    pickle.dump(custom_w2v, handle, protocol=pickle.HIGHEST_PROTOCOL)

# evaluate the model
accr3 = model3d.evaluate(x_test_padded_docs, y_test)
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
modeld.save("C:/Users/Pc/Desktop/proje-1/deneme_google.h5")

# GloVe Word2Vec modeli
model2d.save("C:/Users/Pc/Desktop/proje-1/deneme_glove.h5")

# Eğitim verileriyle eğitilmiş model
model3d.save("C:/Users/Pc/Desktop/proje-1/deneme_custom.h5")

