
from __future__ import print_function

import csv
import os
import numpy as np
import keras
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input
from keras.models import Model
from sklearn import metrics

max_words = 10000
EMBEDDING_DIM = 100
MAX_HEAD_LENGTH = 30
MAX_ART_LENGTH = 200

train_headlines = []
train_articles = []
train_labels = []
all_texts = []

#hash to map labels to integers
label_dict = {'agree': 0, 'disagree':1, 'discuss':2, 'unrelated':3}
with open('data/training_data.csv', 'rt') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:  
        headline, article, label = row
        train_headlines.append(headline)
        train_articles.append(article)
        all_texts.append(headline + ' ' + article)
        num_label = label_dict[label]
        train_labels.append(num_label)
        
test_headlines = []
test_articles = []
test_labels = []

with open('data/dev_data.csv', 'rt') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:  
        headline, article, label = row
        test_headlines.append(headline)
        test_articles.append(article)
        all_texts.append(headline + ' ' + article)
        num_label = label_dict[label]
        test_labels.append(num_label)
test_labels = keras.utils.to_categorical(np.asarray(test_labels))

tk = Tokenizer(num_words=max_words)
tk.fit_on_texts(all_texts)
word_index = tk.word_index
print('Found %s unique tokens.' % len(word_index))

headline_sequences = tk.texts_to_sequences(train_headlines)
article_sequences = tk.texts_to_sequences(train_articles)

test_headline_sequences = tk.texts_to_sequences(test_headlines)
test_article_sequences = tk.texts_to_sequences(test_articles)

test_headline_data = pad_sequences(test_headline_sequences, maxlen=MAX_HEAD_LENGTH)
test_article_data = pad_sequences(test_article_sequences, maxlen=MAX_ART_LENGTH)

GLOVE_DIR = '../project/embeddings'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

headline_data = pad_sequences(headline_sequences, maxlen=MAX_HEAD_LENGTH)
article_data = pad_sequences(article_sequences, maxlen=MAX_ART_LENGTH)
train_labels = keras.utils.to_categorical(np.asarray(train_labels))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

headline_embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_HEAD_LENGTH,
                            trainable=True)

article_embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_ART_LENGTH,
                            trainable=True)

headline_sequence_input = Input(shape=(MAX_HEAD_LENGTH,), dtype='int32')
embedded_headline = headline_embedding_layer(headline_sequence_input)
article_sequence_input = Input(shape=(MAX_ART_LENGTH,), dtype='int32')
embedded_article = article_embedding_layer(article_sequence_input)

headline_lstm = Bidirectional(LSTM(64, return_sequences = False))
article_lstm = Bidirectional(LSTM(64, return_sequences = False))

encoded_headline = headline_lstm(embedded_headline)
encoded_headline = Dropout(.2)(encoded_headline)

encoded_article = article_lstm(embedded_article)
encoded_article = Dropout(.2)(encoded_article)

merged_vector = keras.layers.concatenate([encoded_article, encoded_headline], axis = -1)
merged_vector = Dense(64, activation = 'relu')(merged_vector)
merged_vector = Dropout(.2)(merged_vector)

preds = Dense(4, activation='softmax')(merged_vector)

model = Model(inputs=[article_sequence_input, headline_sequence_input], outputs=preds)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([article_data, headline_data], train_labels, validation_data = ([test_article_data, test_headline_data], test_labels), epochs=12)

model_name = 'Bidirection_LSTM'
model.save(model_name + '.h5')

prediction = model.predict([test_article_data, test_headline_data], batch_size = 32, verbose = 1)

pred_labels = [np.argmax(pred) for pred in prediction]
t_labels = [np.argmax(label) for label in test_labels]
acc = metrics.accuracy_score(pred_labels, t_labels)
precision = metrics.precision_score(pred_labels, t_labels, average = 'micro')
recall = metrics.recall_score(pred_labels, t_labels, average = 'micro')
f1 = metrics.f1_score(pred_labels, t_labels, average = 'micro')
conf_matrix = metrics.confusion_matrix(pred_labels, t_labels)
print('overall testing accuracy: ', acc)

with open('output/'+model_name+'_output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    for i in range(len(test_labels)):
        output_line = [test_headlines[i], test_articles[i][:500], str(t_labels[i]), str(pred_labels[i])]
        writer.writerow(output_line)

t1_pred_labels =  [0 if x == 3 else 1 for x in pred_labels]
t1_test_labels =  [0 if x == 3 else 1 for x in t_labels]
t1_acc = metrics.accuracy_score(t1_pred_labels, t1_test_labels)
t2_pred_labels = []
t2_test_labels = []
for i in range(len(t_labels)):
    if t_labels[i] != 3:
        t2_pred_labels.append(pred_labels[i])
        t2_test_labels.append(t_labels[i])
t2_acc = metrics.accuracy_score(t2_pred_labels, t2_test_labels)

final_acc = t1_acc*.25+t2_acc*.75

print('evaluation score: ', final_acc)

