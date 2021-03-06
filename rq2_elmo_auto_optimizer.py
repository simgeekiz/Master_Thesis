import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda, \
                         Activation, CuDNNLSTM, Dropout
from keras.optimizers import RMSprop, Adam, Adamax
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.layers.merge import add
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

from src.keras_bert import initialize_vars

# Custom
from src.callbacks import PlotCurves
from src.eval_metrics_seq import f1_macro, f1_micro
from src.load_data import load_data

sess = tf.compat.v1.Session()
K.set_session(sess)

with tf.device("gpu:0"):
    print("GPU enabled")

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'

elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=True)
print("ELMo model loaded")


def ELMoEmbeddingStack(x):
    """
    ELMo takes list of sentences (as strings) and returns list of vectors.
    Thus when an article is given to elmo(), it returns a vector for each sentence.

    >> elmo(['I saw a cat.', 'There was also a dog.'])
    [<1024>, <1024>]
    """
    embeds = []
    for art in tf.unstack(tf.transpose(x, (1, 0))):
        embeds.append(elmo(tf.squeeze(tf.cast(art, tf.string)), signature='default', as_dict=True)['default'])
    return tf.stack(embeds, 1)


################# MODELS #################

def build_model_0(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_1(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_2(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
    x = Dropout(0.4)(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_3(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_4(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_5(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)
    x = Dropout(0.4)(x)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_6(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
    x = Dropout(0.4)(x)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_7(max_len):

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_8(max_len):

    def residual(x):
        x_res = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)

    x = residual(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_9(max_len):

    def residual(x):
        x_res = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)

    x = residual(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_10(max_len):

    def residual(x):
        x_res= Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)

    x = residual(x)

    x = Dropout(0.4)(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_11(max_len):

    def residual(x):
        x_res = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)

    x = residual(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_12(max_len):

    def residual(x):
        x_res = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)

    x = residual(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_13(max_len):

    def residual(x):
        x_res = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x_res, x])
        return x

    input_text = Input(shape=(max_len,), dtype='string')

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)

    x = Dense(256, activation='relu')(embedding)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)

    x = residual(x)

    pred = TimeDistributed(Dense(2, activation='softmax'))(x)

    return Model(inputs=[input_text], outputs=pred)


################# UTILS #################


def get_input(data_, max_len, one_hot=False):

    def normalize(text):
        return text.replace('\n', '').strip()

    X = []
    for article in data_:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(normalize(article['sentences'][i]['sentence']))
            except:
                new_seq.append("ENDPAD")
        X.append(new_seq)

    if one_hot:
        y = [[sent['label'] for sent in article['sentences']] for article in data_]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)
        y = [[to_categorical(lab, num_classes=2) for lab in sent] for sent in y]
    else:
        y = [sent['label'] for article in data_ for sent in article['sentences']]

    return np.array(X), np.array(y)


def get_scores(model, data_, batch_size, max_len, results_file, print_out=False):

    def unpad(X, y_preds):
        y_unpad = []
        for ai, art in enumerate(X):
            for si, sent in enumerate(art):
                if sent != 'ENDPAD':
                    y_unpad.append(y_preds[ai][si])
        return y_unpad

    X, y_true = get_input(data_, max_len, one_hot=False)

    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = unpad(X, y_preds)
    y_preds = np.argmax(y_preds, axis=1)

    clsrpt = classification_report(y_true, y_preds)
    sf1 = scikit_f1_score(y_true, y_preds)
    sfm = scikit_f1_score(y_true, y_preds, average='macro')

    if print_out:
        print(clsrpt)
        print('\nScikit_F1_Macro:', sfm)
        print('\nScikit_F1_1:', sf1)

    if results_file:
        with open(results_file, 'a') as f:
            f.write('\n' + clsrpt + '\nF1_Macro: ' + str(sfm) + '\nF1_1: ' + str(sf1) + '\n\n')

    return sfm


################# MAIN #################

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print(("############################\n"
               "ERROR: Give the arguments; model_no, optimizer, learning_rate"
               "\n############################"))
        sys.exit()

    #### INIT PARAMS ####

    max_len = 58
    batch_size = 2

    model_no = sys.argv[1]
    optimizer_name = sys.argv[2]
    lr = float(sys.argv[3])

    loss = 'binary_crossentropy'
    metrics = ['acc', f1_macro, f1_micro]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()
    train_data = [art for art in train_data if len(art['sentences']) > 1]

    #     Limit for testing the pipeline
    #     train_data = train_data[:4]
    #     valid_data = valid_data[:4]
    #     test_data = test_data[:4]

    X_tra, y_tra = get_input(train_data, max_len, one_hot=True)
    X_val, y_val = get_input(valid_data, max_len, one_hot=True)

    del train_data

    #### CONFIGURE MODEL ####

    model = getattr(sys.modules[__name__], 'build_model_' + model_no)(max_len)

    if optimizer_name == 'adam':
        optimizer = Adam(lr=lr)

    elif optimizer_name == 'adamax':
        optimizer = Adamax(lr=lr)

    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(lr=lr)

    else:
        print(("############################\nERROR: Unknown optimizer name!\n############################"))
        sys.exit()

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model_name = 'Optimized_RQ2_elmo' + \
                 '_model_' + model_no + \
                 '_maxlen_' + str(max_len) + \
                 '_' + optimizer_name + \
                 '_lr_' + str(lr) + \
                 '_lrreduction' + \
                 '_loss_' + loss + \
                 '_onehot' + \
                 '_softmax'

    model_main = './Model/' + model_name.split('model')[0] + 'model/'
    model_dir = os.path.join(model_main, model_name)
    score_file = os.path.join(model_main, 'model_performances.csv')
    results_file = os.path.join(model_dir, 'model_results_file.txt')

    print("\n----------------------------------\nFitting the model: ", model_name)

    # Instantiate variables
    initialize_vars(sess)

    model.fit(X_tra, y_tra,
              epochs=50,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[
                  PlotCurves(model_name=model_name, model_dir=model_dir,
                             plt_show=False, jnote=False),
                  ReduceLROnPlateau(monitor='val_f1_macro', patience=3,
                                    factor=0.1, min_lr=0.00001),
                  EarlyStopping(monitor='val_f1_macro', min_delta=0,
                                patience=10, mode='max')
              ])

    best_model = load_model(os.path.join(model_dir, model_name + '_best_f1_macro_model.h5'),
                            custom_objects={'elmo': elmo, 'tf': tf,
                                            'f1_macro': f1_macro,
                                            'f1_micro': f1_micro})

    print("Evaluating the model...")

    with open(results_file, 'w') as f:
        f.write('\n---------------- Validation ----------------\n')

    val_f1 = get_scores(best_model, valid_data, batch_size, max_len, results_file)

    with open(results_file, 'a') as f:
        f.write('\n---------------- Test ----------------\n')

    test_f1 = get_scores(best_model, test_data, batch_size, max_len, results_file)

    if not os.path.exists(score_file):
        with open(score_file, 'w') as scrf:
            scrf.write("model_name,val_f1_macro,test_f1_macro\n")

    with open(score_file, 'a') as scrf:
        scrf.write(
            '\n' + \
            model_name + ',' + \
            str(val_f1) + ',' + \
            str(test_f1) + '\n'
        )

    print("Finished:", model_name)
