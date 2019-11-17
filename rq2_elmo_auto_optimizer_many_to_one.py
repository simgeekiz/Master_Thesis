
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda, \
                         Activation, CuDNNLSTM, SpatialDropout1D, Dropout, BatchNormalization,\
                         GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop, Adam, Adamax, SGD
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

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMoEmbeddingStack(x):
    """
    ELMo takes list of sentences (as strings) and returns list of vectors.
    Thus when an article is given to elmo(), it returns a vector for each sentence.
    
    >> elmo(['I saw a cat.', 'There was also a dog.'])
    [<1024>, <1024>]
    """
    embeds = []
    for art in tf.unstack(tf.transpose(x, (1, 0))):
        embeds.append(elmo(tf.squeeze(tf.cast(art, tf.string)), signature="default", as_dict=True)["default"])
    return tf.stack(embeds, 1)


################# MODELS #################

def build_model_0(ww, n_tags):
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")
    
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
                      
    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

# def build_model_1(ww, n_tags):
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")
    
#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)

#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)

# def build_model_2(ww, n_tags):
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")
    
#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
#     x = Dropout(0.2)(x)
                      
#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)

def build_model_3(ww, n_tags):
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")
    
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)
    
    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

# def build_model_4(ww, n_tags):
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")

#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
#     x = Activation('relu')(x)
    
#     x = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)

#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)

# def build_model_5(ww, n_tags):
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")

#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
#     x = Dropout(0.4)(x)
    
#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)


def build_model_6(ww, n_tags):
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

# def build_model_7(ww, n_tags):
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")

#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)
#     x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
    
#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)

def build_model_8(ww, n_tags):
    
    def residual(x):
        x_res = x
        
        x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
        x = add([x, x_res])
        return x
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(embedding)
    x = residual(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_9(ww, n_tags):
    
    def residual(x):
        x_res = x
        
        x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = add([x, x_res])
        return x
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
    x = residual(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

# def build_model_10(ww, n_tags):
    
#     def residual(x):
#         x_res = x
        
#         x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
#         x = add([x, x_res])
#         return x
    
#     inp_size = 2*ww+1
#     input_text = Input(shape=(inp_size,), dtype="string")
#     embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
#     x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(embedding)
#     x = residual(x)
#     x = Dropout(0.4)(x)

#     pred = LSTM(n_tags, activation="sigmoid")(x)

#     return Model(inputs=[input_text], outputs=pred)

def build_model_11(ww, n_tags):
    
    def residual(x):
        x_res = x
        
        x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
        x = add([x, x_res])
        return x
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)
    # Note: bir alttaki satirda x yerine embedding yazilmisti.
    x = Bidirectional(CuDNNLSTM(units=128, return_sequences=True))(x)
    x = residual(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

def build_model_12(ww, n_tags):
    
    def residual(x):
        x_res = x

        x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
        x = add([x, x_res])
        return x

    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")

    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)
    
    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)
                                
    x = Bidirectional(CuDNNLSTM(units=256, return_sequences=True))(x)
    x = residual(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)

    return Model(inputs=[input_text], outputs=pred)

def build_model_13(ww, n_tags):
    
    def residual(x):
        x_res = x
        
        x = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)

        x = add([x, x_res])
        return x
    
    inp_size = 2*ww+1
    input_text = Input(shape=(inp_size,), dtype="string")
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, inp_size, 1024))(input_text)    
    x = Dense(256, activation='relu')(embedding)
    x = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)

    x = residual(x)

    pred = LSTM(n_tags, activation="sigmoid")(x)
    
    return Model(inputs=[input_text], outputs=pred)

################# UTILS #################


def get_input(data_, ww, n_tags, batch_size, is_test=False, limit=None):
    
    def normalize(text):
        return text.replace('\n', '').strip()
    
    padding_sent = {
        'sentence': 'ENDPAD',
        'label': 0
    }

    X = []
    y = []

    for article in data_:
        sent_objs = article['sentences']
        
        for si, sentence in enumerate(sent_objs):
            sequence = []

            # Prev
            for i in reversed(range(ww)):
                sequence.append(normalize(sent_objs[si-i-1]['sentence']) 
                                if si-i-1 >= 0 
                                else padding_sent['sentence'])

            # Curr
            sequence.append(normalize(sent_objs[si]['sentence']))

            # Next
            for i in range(ww):
                sequence.append(normalize(sent_objs[si+i+1]['sentence'] )
                                if si+i+1 < len(article['sentences']) 
                                else padding_sent['sentence'])

            X.append(sequence)
            y.append(sent_objs[si]['label']
                     if is_test 
                     else to_categorical(sent_objs[si]['label'], num_classes=n_tags))
            
    # limit data if not an even number when batch_size=2
    if not limit:
        limit = len(X) if len(X)%batch_size == 0 else len(X)-len(X)%batch_size
        X = X[:limit]
        y = y[:limit]

    return np.array(X), np.array(y)

def get_scores(model, data_, batch_size, ww, n_tags, results_file, print_out=False):
    
    X, y_true = get_input(data_, ww, n_tags, batch_size, is_test=True, limit=None)
    
    y_preds = model.predict(X, batch_size=batch_size)
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

    #### INIT PARAMS ####

    ww = 3
    n_tags = 2
    batch_size = 32

    build_models_functions = {
#         'model_0': build_model_0(ww, n_tags),
#         'model_1': build_model_1(ww, n_tags),
#         'model_2': build_model_2(ww, n_tags),
#         'model_3': build_model_3(ww, n_tags),
#         'model_4': build_model_4(ww, n_tags),
#         'model_5': build_model_5(ww, n_tags),
#         'model_6': build_model_6(ww, n_tags),
#         'model_7': build_model_7(ww, n_tags),
#         'model_8': build_model_8(ww, n_tags),
#         'model_9': build_model_9(ww, n_tags),
#         'model_10': build_model_10(ww, n_tags),
#         'model_11': build_model_11(ww, n_tags),
#         'model_12': build_model_12(ww, n_tags),
#         'model_13': build_model_13(ww, n_tags),
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.001, 0.0005]},
#         {'name': 'adamax',
#          'lro': [0.001, 0.0001]},
        {'name': 'rmsprop',
         'lro': [0.001, 0.0005]},
    ]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()

#     Limit for testing the pipeline
#     train_data = train_data[:4]
#     valid_data = valid_data[:4]
#     test_data = test_data[:4]

    X_tra, y_tra = get_input(train_data, ww, n_tags, batch_size, is_test=False, limit=None)
    X_val, y_val = get_input(valid_data, ww, n_tags, batch_size, is_test=False, limit=None)
  
    del train_data

    model = None
    optimizer = None

    loss = 'binary_crossentropy'
    metrics = ['acc', f1_macro, f1_micro]

    for fname, func in build_models_functions.items():

        print("\n----------------------------------\n")
        print("Starting model:", fname)

        for opco in optimizer_configs:

            optimizer_name = opco['name']

            print("Testing optimizer:", optimizer_name)

            for lr in opco['lro']:

                print("Learning rate:", str(lr))

                if optimizer:
                    del optimizer

                if optimizer_name == 'adam':
                    optimizer = Adam(lr=lr)

                elif optimizer_name == 'adamax':
                    optimizer = Adamax(lr=lr)

                elif optimizer_name == 'rmsprop':
                    optimizer = RMSprop(lr=lr)

                if model:
                    del model
                model = func

                model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                model_name = 'Optimized_RQ2_elmo_many_to_one' + \
                             '_' + fname + \
                             '_ww_' + str(ww) + \
                             '_' + optimizer_name +  \
                             '_lr_' + str(lr) +  \
                             '_lrreduction' + \
                             '_loss_' + loss 
                            
                model_main = './Model/' + model_name.split('model')[0] + 'model/'
                model_dir = os.path.join(model_main, model_name)
                score_file = os.path.join(model_main, 'model_performances.csv')
                results_file = os.path.join(model_dir, 'model_results_file.txt')

                print("Fitting the model...")

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
                                        custom_objects={'elmo':elmo, 'tf':tf,
                                                        'f1_macro':f1_macro,
                                                        'f1_micro':f1_micro})

                print("Evaluating the model...")

                with open(results_file, 'w') as f:
                    f.write('\n---------------- Validation ----------------\n')

                val_f1 = get_scores(best_model, valid_data, batch_size, ww, n_tags, results_file, print_out=False)

                with open(results_file, 'a') as f:
                    f.write('\n---------------- Test ----------------\n')

                test_f1 = get_scores(best_model, test_data, batch_size, ww, n_tags, results_file, print_out=False)

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
