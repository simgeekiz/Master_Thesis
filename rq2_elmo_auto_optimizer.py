
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.models import Model, Input, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from keras.optimizers import RMSprop, Adam, Adamax, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.layers.merge import add
from keras.utils import to_categorical

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
    embeds = []
    for art in tf.unstack(tf.transpose(x, (1, 0))):
        embeds.append(elmo(tf.squeeze(tf.cast(art, tf.string)), signature="default", as_dict=True)["default"])
    return tf.stack(embeds, 1)

################# MODELS #################

def build_model_0(max_len, n_tags):
    
    def residual(x):
        x_res = x
        
        x = Bidirectional(LSTM(units=128, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_res])
        return x
    
    input_text = Input(shape=(max_len,), dtype="string")
    
    embedding = Lambda(ELMoEmbeddingStack, output_shape=(None, None, max_len, 1024))(input_text)
    
    x = Dense(512, activation='relu')(embedding)
    
    x = Dense(256, activation='relu')(x)
    
    x = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
    x = residual(x)

    pred = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

    return Model(inputs=[input_text], outputs=pred)


################# UTILS #################


def get_input(data_, max_len, n_tags, is_test=False, limit=None):
    
    # limit data if not an even number when batch_size=2
    if not limit:
        limit = len(data_) if len(data_)%2 == 0 else len(data_)-1

    data_ = data_[:limit]
    
    X = []
    for article in data_:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(article['sentences'][i]['sentence'].replace('\n', '').strip())
            except:
                new_seq.append("ENDPAD")
        X.append(new_seq)
    
    if not is_test: 
        y = [[sent['label'] for sent in article['sentences']] for article in data_]
        y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)
        y = [[to_categorical(lab, num_classes=n_tags) for lab in sent] for sent in y]
    else:
        y = [sent['label'] for article in data_ for sent in article['sentences']]

    return np.array(X), np.array(y)

def get_scores(model, data_, batch_size, max_len, n_tags, results_file, print_out=False):
    
    def unpad(X, y_preds):
        y_unpad = []
        for ai, art in enumerate(X):
            for si, sent in enumerate(art):
                if sent != 'ENDPAD':
                    y_unpad.append(y_preds[ai][si])
        return y_unpad
    
    X, y = get_input(data_, max_len, n_tags, True)
    
    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = unpad(X, y_preds)
    y_preds = np.argmax(y_preds, axis=1)
    
    clsrpt = classification_report(y, y_preds)
    sfm = scikit_f1_score(y, y_preds, average='macro')
    
    if print_out:
        print(clsrpt)
        print('\nScikit_F1_Macro:', sfm)

    if results_file:
        with open(results_file, 'a') as f:
            f.write('\n' + clsrpt + '\n' + str(sfm) + '\n')
            
    return sfm

################# MAIN #################

if __name__ == '__main__':

    #### INIT PARAMS ####

    max_len = 60
    n_tags = 2
    batch_size = 2

    build_models_functions = {
        'model_0': build_model_0(max_len, n_tags),
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.001, 0.0001]},
        {'name': 'adamax',
         'lro': [0.001, 0.0001]},
        {'name': 'rmsprop',
         'lro': [0.001, 0.0001]},
    ]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()

#     Limit for testing the pipeline
#     train_data = train_data[:1]
#     valid_data = valid_data[:1]
#     test_data = test_data[:1]

    X_tra, y_tra = get_input(train_data, max_len, n_tags, False)
    X_val, y_val = get_input(valid_data, max_len, n_tags, False)

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

                model_name = 'Optimized_RQ2_elmo' + \
                             '_' + fname + \
                             '_' + optimizer_name +  \
                             '_lr_' + str(lr) +  \
                             '_lrreduction' + \
                             '_loss_' + loss + \
                             '_maxlen_' + str(max_len)

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
