import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input
from keras.layers import Dense, Lambda
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

# Custom
from src.callbacks import PlotCurves
from src.eval_metrics_seq import f1_macro, f1_micro 
from src.load_data import load_data

sess = tf.Session()
K.set_session(sess)

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMoEmbeddingStack(x):
    embeds = []
    for art in tf.unstack(tf.transpose(x, (1, 0))):
        embeds.append(elmo(tf.squeeze(tf.cast(art, tf.string)), signature="default", as_dict=True)["default"])
    return tf.stack(embeds, 1)

def build_rnn_model_0(max_len, n_tags):
    
    input_text = Input(shape=(max_len,), dtype="string")
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)
    dns = Dense(512, activation='relu')(embedding)
    dns = Dense(256, activation='relu')(dns)
    x = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(dns)
    x_rnn = Bidirectional(LSTM(units=128, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)
 
    return Model(input_text, outputs=out)

def get_input(data_, max_len, n_tags, is_test=False, limit=None):
    
    X = []
    for article in data_:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(article['sentences'][i]['sentence'].replace('\n', '').strip().lower())
            except:
                new_seq.append("ENDPAD")
        X.append(new_seq)
    
    if not is_test: 
        y = [[sent['label'] for sent in article['sentences']] for article in data_]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)
        y = [[to_categorical(lab, num_classes=n_tags) for lab in sent] for sent in y]
    else:
        y = [sent['label'] for article in data_ for sent in article['sentences']]
    
    if not limit:
        limit = len(X) if len(X)%2 == 0 else len(X)-1

    return np.array(X)[:limit], np.array(y)[:limit]

def get_scores(model, data_, batch_size, max_len, n_tags, results_file):
    
    X, y = get_input(data_, max_len, n_tags, True)
    
    def unpad(X, y):
        y_unpad = []
        for ai, art in enumerate(X):
            for si, sent in enumerate(art):
                if sent != 'ENDPAD':
                    y_unpad.append(y[ai][si])
        return y_unpad
    
    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = [[np.argmax(y) for y in art] for art in y_preds]
    y_preds = unpad(X, y_preds)
    
    clsrpt = classification_report(y, y_preds)
    sfm = scikit_f1_score(y, y_preds, average='macro')

    if results_file:
        with open(results_file, 'a') as f:
            f.write('\n' + clsrpt + '\n' + str(sfm) + '\n')
            
    return sfm
