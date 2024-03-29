
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import Dense, Lambda, Activation, Conv1D, \
                         MaxPooling1D, Flatten, Reshape, \
                         BatchNormalization, Dropout, Bidirectional, LSTM , CuDNNGRU, GRU
from keras.optimizers import RMSprop, Adam, Adamax
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.layers.merge import add
from keras.utils import to_categorical

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

from src.keras_bert import initialize_vars

# Custom
from src.callbacks import PlotCurves
from src.eval_metrics import f1_macro, f1_micro
from src.load_data import load_data

sess = tf.compat.v1.Session()
K.set_session(sess)

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]

################# MODELS #################

def build_model_83(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_84(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(256))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_86(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(GRU(512, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_87(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(512))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_91(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(512))(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_92(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(512))(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_93(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(512))(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_94(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(256))(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_95(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(128))(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_96(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(128))(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_97(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(128))(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_98(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(512, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(256))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)

def build_model_99(n_tags):
    
    elmo_input_layer = Input(shape=(None, ), dtype="string")
    x = Lambda(ELMoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    x = Bidirectional(CuDNNGRU(256, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(128))(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    return Model(inputs=elmo_input_layer, outputs=x)


################# UTILS #################


def get_input(data_, n_tags, is_test=False):

    X = np.array([sentence['sentence'].replace('\n', '').strip().lower()
                  for article in data_
                  for sentence in article['sentences']])

    y = np.array([sentence['label']
                  for article in data_
                  for sentence in article['sentences']])

    if not is_test:
        y = to_categorical(y, num_classes=n_tags)

    return X, y

def get_scores(model, data_, batch_size, n_tags, results_file, print_out=False):

    X, y = get_input(data_, n_tags, True)

    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = np.argmax(y_preds, axis=1)

    clsrpt = classification_report(y, y_preds)
    sfm = scikit_f1_score(y, y_preds, average='macro')

    if print_out:
        print(clsrpt)
        print('\nScikit_F1_Macro:', sfm)

    if results_file:
        with open(results_file, 'a') as f:
            f.write('\n' + clsrpt + '\n' + str(sfm) + '\n\n')

    return sfm

################# MAIN #################

if __name__ == '__main__':

    #### INIT PARAMS ####

    n_tags = 2
    batch_size = 32

    build_models_functions = {
#         'model_81': build_model_81(n_tags),
#         'model_82': build_model_82(n_tags),
        'model_83': build_model_83(n_tags),
        'model_84': build_model_84(n_tags),
#         'model_85': build_model_85(n_tags),
#         'model_86': build_model_86(n_tags),
        'model_87': build_model_87(n_tags),
#         'model_88': build_model_88(n_tags),
#         'model_89': build_model_89(n_tags),
#         'model_50': build_model_50(n_tags),
        'model_91': build_model_91(n_tags),
        'model_92': build_model_92(n_tags),
        'model_93': build_model_93(n_tags),
#         'model_94': build_model_94(n_tags),
#         'model_95': build_model_95(n_tags),
#         'model_96': build_model_96(n_tags),
        'model_97': build_model_97(n_tags),
#         'model_98': build_model_98(n_tags),
        'model_99': build_model_99(n_tags),
        
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.001]},
        {'name': 'rmsprop',
         'lro': [0.001]},
    ]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()

#     Limit for testing the pipeline
#     train_data = train_data[:1]
#     valid_data = valid_data[:1]
#     test_data = test_data[:1]

    X_tra, y_tra = get_input(train_data, n_tags, False)
    X_val, y_val = get_input(valid_data, n_tags, False)

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

                model_name = 'Optimized_RQ1_aion_elmo' + \
                             '_' + fname + \
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

                val_f1 = get_scores(best_model, valid_data, batch_size, n_tags, results_file)

                with open(results_file, 'a') as f:
                    f.write('\n---------------- Test ----------------\n')

                test_f1 = get_scores(best_model, test_data, batch_size, n_tags, results_file)

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
