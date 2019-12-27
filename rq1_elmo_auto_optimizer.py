import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import Dense, Lambda, Activation, Conv1D, \
                         MaxPooling1D, Flatten, Reshape, \
                         BatchNormalization, Dropout
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

with tf.device("gpu:0"):
    print("GPU enabled")

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'

elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=True)
print("ELMo model loaded")

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature='default', as_dict=True)['default']


################# MODELS #################

def build_model_0():
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(512, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_1():
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_2():

    def residual(x):
        x_res = Dense(256, kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_3():

    def residual(x):
        x_res = Dense(256, kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x_res = Dense(256, kernel_regularizer=l2(0.001))(x_res)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_4():

    def residual(x):
        x_res = Dense(512, kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(512, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_5():

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    embedding = Reshape((1024, 1))(embedding)

    x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_6():

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    embedding = Reshape((1024, 1))(embedding)

    x = Conv1D(128, 5, padding='same', kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(128, 5, padding='same', kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_7():

    def residual(x):
        x_res = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x_res = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(x_res)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    embedding = Reshape((1024, 1))(embedding)

    x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_8():

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(512, kernel_regularizer=l2(0.001))(embedding)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_9():

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_10():

    def residual(x):
        x_res = Dense(256, kernel_regularizer=l2(0.001))(x)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Dropout(0.3)(x_res)

        x_res = Dense(256, kernel_regularizer=l2(0.001))(x_res)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Dropout(0.3)(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_11():

    def residual(x):
        x_res = Dense(256, kernel_regularizer=l2(0.001))(x)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)

        x_res = Dense(256, kernel_regularizer=l2(0.001))(x_res)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_12():

    def residual(x):
        x_res = Dense(512, kernel_regularizer=l2(0.001))(x)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Dropout(0.5)(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(512, kernel_regularizer=l2(0.001))(embedding)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_13():

    def residual(x):
        x_res = Dense(256, kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_14():

    def residual(x):
        x_res = Dense(128, kernel_regularizer=l2(0.001))(x)
        x_res = Activation('relu')(x_res)

        x = add([x_res, x])
        return x

    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)

    x = Dense(128, kernel_regularizer=l2(0.001))(embedding)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)

    pred = Dense(2, activation='softmax')(x)

    return Model(inputs=[input_text], outputs=pred)


################# UTILS #################


def get_input(data_, one_hot=False):

    X = np.array([sentence['sentence'].replace('\n', '').strip()
                  for article in data_
                  for sentence in article['sentences']])

    y = np.array([sentence['label']
                  for article in data_
                  for sentence in article['sentences']])

    if one_hot:
        y = to_categorical(y, num_classes=2)

    return X, y


def get_scores(model, data_, batch_size, results_file, print_out=False):

    X, y_true = get_input(data_, one_hot=False)

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

    batch_size = 32

    build_models_functions = {
        'model_0': build_model_0(),
        'model_1': build_model_1(),
        'model_2': build_model_2(),
        'model_3': build_model_3(),
        'model_4': build_model_4(),
        'model_5': build_model_5(),
        'model_6': build_model_6(),
        'model_7': build_model_7(),
        'model_8': build_model_8(),
        'model_9': build_model_9(),
        'model_10': build_model_10(),
        'model_11': build_model_11(),
        'model_12': build_model_12(),
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.01, 0.001, 0.005]},
        {'name': 'rmsprop',
         'lro': [0.01, 0.001, 0.005]},
        # {'name': 'adamax',
        #  'lro': [0.01, 0.001]},#0.01,
    ]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()

    # Limit for testing the pipeline
    # train_data = train_data[:1]
    # valid_data = valid_data[:1]
    # test_data = test_data[:1]

    X_tra, y_tra = get_input(train_data, one_hot=True)
    X_val, y_val = get_input(valid_data, one_hot=True)

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

                model_name = 'Optimized_RQ1_elmo' + \
                             '_' + fname + \
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
                                        custom_objects={'elmo': elmo, 'tf': tf,
                                                        'f1_macro': f1_macro,
                                                        'f1_micro': f1_micro})

                print("Evaluating the model...")

                with open(results_file, 'w') as f:
                    f.write('\n---------------- Validation ----------------\n')

                val_f1 = get_scores(best_model, valid_data, batch_size, results_file)

                with open(results_file, 'a') as f:
                    f.write('\n---------------- Test ----------------\n')

                test_f1 = get_scores(best_model, test_data, batch_size, results_file)

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
