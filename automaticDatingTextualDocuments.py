from __future__ import print_function
import numpy as np
import os

np.random.seed(1337)  # for reproducibility

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from sklearn.cross_validation import StratifiedKFold
from nltk import tokenize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.callbacks import *
from keras.utils.visualize_util import plot
from keras.layers import merge
from theano import *
from math import sqrt

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init((input_shape[-1],))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

	self.att = weights
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

def get_word_att(model, batch):
    get_word_att = theano.function([model.layers[0].input], model.layers[1].layer.layers[4].att)
    word_att = get_word_att(batch)
    return word_att

def get_sent_att(model, batch):
    get_sent_att = theano.function([model.layers[0].input], model.layers[4].att)
    sent_att = get_sent_att(batch)
    return sent_att

##########   DATASET SECTION  #########################

# Set parameters:
max_features = 40000
maxlen = 37
maxsents =3
batch_size = 32
embedding_dims = 300
nb_epoch = 350
validation_split = 0.25
pretrained_embeddings = True
lstm_output_size = 50

print('Loading data...')

#get training dates semeval

f= open("data/training08T2.txt", 'r')

dates=[]
dates_dec=[]
dates_lin=[]
c=1
interval=""

lines=f.readlines()
dates_lines1=[]
vec_dates=[]

Initial_D=[] #first date of interval
Final_D=[] #last date of the interval

while c < 22007:
        dates_lines1.append(lines[c])
        c +=7


for line in dates_lines1:
        pos = 0
        conj_dates = line.split(" ")
        while pos < len(conj_dates):

                if conj_dates[pos][0] == "y":
                        interval = str(conj_dates[pos])
                        pos += 1
                else:
                        pos += 1
        interval = interval.split("\"")
        interval=interval[1].split("-")
        Initial_D.append(interval[0])
        Final_D.append(interval[1])
        final_date = int((int(interval[0])+int(interval[1]))/2)
        dates.append(final_date-1699)
	dates_dec.append((final_date-1699)/10)
	dates_lin.append(float(final_date-1699)/312)
        c += 7

print(min(dates))
#print(len(dates))

f.close()

c=1
dates_lines2=[]

f= open("data/training12T2.txt", 'r')

while c < 7167:
        dates_lines2.append(lines[c])
        c +=7


for line in dates_lines2:
        pos = 0
        conj_dates = line.split(" ")
        while pos < len(conj_dates):

                if conj_dates[pos][0] == "y":
                        #print(conj_dates[pos])
                        #print(conj_dates[pos][0])
                        interval = str(conj_dates[pos])
                        pos += 1
                else:
                        pos += 1
        interval = interval.split("\"")
        interval=interval[1].split("-")
        Initial_D.append(interval[0])
        Final_D.append(interval[1])
        final_date = int((int(interval[0])+int(interval[1]))/2)
        dates.append(final_date-1699)
	dates_dec.append((final_date-1699)/10)
	dates_lin.append(float(final_date-1699)/312)
        c += 7

print(dates)
print(max(dates))

f.close()

#get training frases semeval

f= open("data/training08T2.txt", 'r')

texts=[]

c=4
lines=f.readlines()
for line in lines:

    while c < 22007:
            text = lines[c]
            text.replace('\n', ' ')
            text.rstrip()
            texts.append(text.decode("ascii", "ignore").encode("utf8"))
            c += 7

f.close()

c=4

f= open("data/training12T2.txt", 'r')

lines=f.readlines()
for line in lines:

    while c < 7167:
            text = lines[c]
            text.replace('\n', ' ')
            text.rstrip()
            texts.append(text.decode("ascii", "ignore").encode("utf8"))
            c += 7

f.close()
print(len(texts))


print("data treino minima: " + str(min(dates)))
print("data treino maxima: " + str(max(dates)))
print(sum(x > 210 for x in dates))

#get test dates semeval

f= open("data/testT2.txt", 'r')

dates_test=[]
dates_dec_test=[]
dates_lin_test=[]
c=1
interval=""

lines=f.readlines()
dates_lines=[]

Initial_D=[] #first date of interval
Final_D=[] #last date of the interval

while c < 7286:
        dates_lines.append(lines[c])
        c +=7


for line in dates_lines:
        pos = 0
        conj_dates = line.split(" ")
        while pos < len(conj_dates):

                if conj_dates[pos][0] == "y":
                        interval = str(conj_dates[pos])
                        pos += 1
                else:
                        pos += 1
        interval = interval.split("\"")
        interval=interval[1].split("-")
        Initial_D.append(interval[0])
        Final_D.append(interval[1])
        final_date = int((int(interval[0])+int(interval[1]))/2)
        dates_test.append(final_date-1699)
	dates_dec_test.append((final_date-1699)/10)
	dates_lin_test.append(float(final_date-1699)/312)
        c += 7

print(dates_test)
print("data teste minima: " + str(min(dates_test)))
print("data teste maxima: " + str(max(dates_test)))


f.close()

#get test frases semeval

f= open("data/testT2.txt", 'r')

texts_test=[]

c=4
lines=f.readlines()
for line in lines:

    while c < 7286:
            text = lines[c]
            text.replace('\n', ' ')
            text.rstrip()
            texts_test.append(text.decode("ascii", "ignore").encode("utf8"))
            c += 7

f.close()

######## END OF DATASET SECTION ##############

n_texts=0
n_sents=0
n_words=0
vec_sent=[]
vec_words=[]

tokenizer_train = Tokenizer(nb_words=max_features)
tokenizer_train.fit_on_texts(texts)
data = np.zeros((len(texts), maxsents, maxlen), dtype='int32')
for i, sentences in enumerate(texts):
    n_texts+=1
    sentences = tokenize.sent_tokenize( sentences )
    for j, sent in enumerate(sentences):
        n_sents+=1
        if j< maxsents:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _ , word in enumerate(wordTokens):
		n_words+=1
                if k < maxlen and tokenizer_train.word_index[word] < max_features:
                    data[i,j,k] = tokenizer_train.word_index[word]
                    k=k+1
	vec_words.append(n_words)
	n_words=0
    vec_sent.append(n_sents)
    n_sents=0


word_index = tokenizer_train.word_index



print('Found %s unique tokens.' % len(word_index))
X_train = np.asarray( data )

dates = np.asarray(dates)
dates_dec = np.asarray(dates_dec)
Y_train_lin = np.asarray (dates_lin)
Y_train = to_categorical(dates)
Y_dec_train = to_categorical(dates_dec)

n_sents=0
n_words=0

data_test = np.zeros((len(texts_test), maxsents, maxlen), dtype='int32')
for i, sentences in enumerate(texts_test):
    n_texts+=1
    sentences = tokenize.sent_tokenize( sentences )
    for j, sent in enumerate(sentences):
	n_sents+=1
        if j< maxsents:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _ , word in enumerate(wordTokens):
		n_words+=1
                if word in tokenizer_train.word_index:
                    if k < maxlen and tokenizer_train.word_index[word] < max_features:
                        data_test[i,j,k] = tokenizer_train.word_index[word]
                        k=k+1
	vec_words.append(n_words)
	n_words=0
    vec_sent.append(n_sents)
    n_sents=0
word_index = tokenizer_train.word_index


X_test = np.asarray( data_test )
Y_test_lin = np.asarray( dates_lin_test )
dates_test = np.asarray(dates_test)
dates_dec_test = np.asarray(dates_dec_test)
Y_test = to_categorical(dates_test, nb_classes=312)
Y_dec_test = to_categorical(dates_dec_test, nb_classes=32)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


embedding_matrix = None
if pretrained_embeddings:
    print('Read word embeddings...')
    embeddings_index = {}
    f = open("glove/glove.6B.300d.txt", 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # prepare embedding matrix
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.rand(max_features, embedding_dims)
    for word, i in word_index.items():
        if i >= nb_words: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

		
#This EarlyStopping values were tuned using the semeval dataset		
callbacks=[
    EarlyStopping(monitor='loss',min_delta=0.0000001, patience=2, verbose=0, mode='auto')
]

print('Build model...')
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
embedding_layer = Embedding(max_features, embedding_dims, input_length=maxlen)
if pretrained_embeddings: embedding_layer = Embedding(max_features, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=True)
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)

# Hierarchical attention model
# First level considers the words of each sentence, and the second level uses the output of 
# the first level to construct a sentence representation to use in the second level

l_lstm = Bidirectional(GRU(lstm_output_size, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(lstm_output_size, activation='sigmoid'))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)
review_input = Input(shape=(maxsents,maxlen), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(lstm_output_size, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(lstm_output_size, activation='sigmoid'))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
postp = Dense(lstm_output_size, activation='softmax')(l_att_sent)

# Approach similar to that of fasttext, corresponding to the simplest branch of the model
sentEmbed = Model(sentence_input, embedded_sequences)
review_fasttext = TimeDistributed(sentEmbed)(review_input)
fasttext = (GlobalAveragePooling2D()(review_fasttext))


# Final model results from the merge of the two approaches
postp = merge( [ postp , fasttext ] , mode='concat', concat_axis = 1 )
postp = Dropout(0.01)(postp)

preds = Dense(Y_train.shape[1], activation='softmax', name='main_output')(postp)
preds_dec= Dense(Y_dec_train.shape[1], activation='softmax', name='output_dec')(postp)
preds_linear = Dense(1, activation='linear', name='output_lin')(postp)

# The loss_weights were tuned using the semeval dataset
model = Model(review_input, [preds, preds_dec, preds_linear])
model.compile(loss={'main_output': 'categorical_crossentropy', 'output_dec': 'categorical_crossentropy', 'output_lin': 'mean_squared_error'}, optimizer='adam', metrics=['accuracy'], loss_weights = [1.0 , 1.0, 0.25])
model.fit(X_train, [Y_train, Y_dec_train, Y_train_lin] , batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, [Y_test, Y_dec_test, Y_test_lin]))#, callbacks=callbacks)



[y, d, l] = model.predict(X_test)
v=0
pred=[]
print(str(len(y[1])))
while v < len(y):
    pred.append(np.argmax(y[v]))
    v+=1
#print(pred)
correct = np.where(Y_test)[1]

very_good=0
good = 0
total = 0

n = 0
while n< len(pred):
    low_correct=correct[n]-10
    high_correct=correct[n]+10
    if pred[n] == correct[n]:
        very_good+=1
        good+=1
        total+=1
    elif pred[n] > low_correct and pred[n] < high_correct :
        good+=1
        total+=1
    else:
        total+=1
    n+=1
semeval_acc=float(good)/total
semeval_precision=float(very_good)/total


print("\n")
print("SEMEVAL_ACC: " + str(semeval_acc))
print("\n")
print("SEMEVAL_PRECISION: " + str(semeval_precision))
print("\n")
print(pred)
print(np.where(Y_test)[1])

MAE=0
RMSE=0

MAE=mean_absolute_error(np.where(Y_test)[1],pred)
RMSE=sqrt(mean_squared_error(np.where(Y_test)[1],pred))

print("\n")
print("Mean Absolute Error: " + str(MAE))
print("\n")
print("Root Mean Squared Error: " + str(RMSE))





