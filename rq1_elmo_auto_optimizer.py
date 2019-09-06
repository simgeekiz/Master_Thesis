
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
from src.eval_metrics import f1_macro, f1_micro 
from src.load_data import load_data

sess = tf.Session()
K.set_session(sess)

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def build_fnn_model_0():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(512, activation='relu')(embedding)
    dense = Dense(256, activation='relu')(dense)
    pred = Dense(2, activation='sigmoid')(dense)

    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_1():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(512, activation='relu')(embedding)
    dense = Dense(256, activation='relu')(dense)
    dense = Dense(128, activation='relu')(dense)
    pred = Dense(2, activation='sigmoid')(dense)

    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_2():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu')(embedding)
    dense = Dense(128, activation='relu')(dense)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_3():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(512, activation='relu')(embedding)
    dense = Dense(128, activation='relu')(dense)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_4():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(512, activation='relu')(embedding)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_5():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu')(embedding)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_6():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(128, activation='relu')(embedding)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_7():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(embedding)
    dense = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(dense)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

def build_fnn_model_8():
    
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
    dense = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(embedding)
    dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(dense)
    pred = Dense(2, activation='sigmoid')(dense)
    return Model(inputs=[input_text], outputs=pred)

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

def get_scores(model, data_, batch_size, n_tags, results_file):
    
    X, y = get_input(data_, n_tags, True)
    
    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = np.argmax(y_preds, axis=1)
    
    clsrpt = classification_report(y, y_preds)
    sfm = scikit_f1_score(y, y_preds, average='macro')

    if results_file:
        with open(results_file, 'a') as f:
            f.write('\n' + clsrpt + '\n' + str(sfm) + '\n\n')
            
    return sfm


if __name__ == '__main__':

    build_models_functions = {
        'fnn_model_0': build_fnn_model_0(),
        'fnn_model_1': build_fnn_model_1(), 
        'fnn_model_2': build_fnn_model_2(),
        'fnn_model_3': build_fnn_model_3(), 
        'fnn_model_4': build_fnn_model_4(), 
        'fnn_model_5': build_fnn_model_5(),
        'fnn_model_6': build_fnn_model_6(),
        'fnn_model_7': build_fnn_model_7(),
        'fnn_model_8': build_fnn_model_8(),
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.005, 0.001, 0.0005, 0.0001]}, # learning rate options
        {'name': 'rmsprop',
         'lro': [0.005, 0.001, 0.0005, 0.0001]},
        {'name': 'sgd',
         'lro': [0.05, 0.01, 0.001, 0.005]},
    ]
    
    n_tags = 2
    batch_size = 32
    
    train_data, valid_data, test_data, _ = load_data()
    
#     Limit for testing the pipeline
#     train_data = train_data[:10]
#     valid_data = valid_data[:10]
#     test_data = test_data[:10]
    
    X_tra, y_tra = get_input(train_data, n_tags, False)
    X_val, y_val = get_input(valid_data, n_tags, False)
    
    del train_data
       
    loss = 'binary_crossentropy'
    metrics = ['acc', f1_macro, f1_micro]
    
    for fname, func in build_models_functions.items():
        
        print("\n----------------------------------\n")
        print("Starting model:", fname)
        
        model = func
        
        for opco in optimizer_configs:
            
            optimizer_name = opco['name']
            
            print("Testing optimizer:", optimizer_name)
        
            for lr in opco['lro']:
                
                print("Learning rate:", str(lr))
            
                if optimizer_name == 'adam':
                    optimizer = Adam(lr=lr)
                    
                elif optimizer_name == 'rmsprop':
                    optimizer = RMSprop(lr=lr)
                    
                elif optimizer_name == 'sgd':
                    optimizer = SGD(lr=lr)
    
                model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                model_name = 'Optimized_RQ1_elmo' + \
                             '_' + fname + \
                             '_' + optimizer_name +  \
                             '_lr_' + str(lr) +  \
                             '_loss_' + loss

                model_main = './Model/' + model_name.split('model')[0] + 'model/'
                model_dir = os.path.join(model_main, model_name)
                score_file = os.path.join(model_main, 'model_performances.csv')
                results_file = os.path.join(model_dir, 'model_results_file.txt')
                
                print("Fitting the model...")

                model.fit(X_tra, y_tra, 
                          epochs=15, 
                          batch_size=batch_size, 
                          validation_data=(X_val, y_val), 
                          callbacks=[
                              PlotCurves(model_name=model_name, model_dir=model_dir, save=False)
                          ])
                
                best_model = load_model(os.path.join(model_dir, model_name + '_best_f1_macro_model.h5'), 
                                        custom_objects={'elmo':elmo, 'tf':tf, 
                                                        'f1_macro':f1_macro, 'f1_micro':f1_micro})
                
                print("Evaluating the model...")
                
                with open(results_file, 'w') as f:
                    f.write('\n---------------- Validation ----------------\n')

                val_f1 = get_scores(best_model, valid_data, batch_size, n_tags, results_file)
                
                with open(results_file, 'a') as f:
                    f.write('\n---------------- Test ----------------\n')
                    
                test_f1 = get_scores(best_model, test_data, batch_size, n_tags, results_file)
                
                if not os.path.exists(score_file):
                    with open(score_file, 'w') as scrf:
                        scrf.write("model_name,val_f1_macro,test_f1_macro")
                
                with open(score_file, 'a') as scrf:
                    scrf.write(
                        '\n' + \
                        model_name + ',' + \
                        str(val_f1) + ',' + \
                        str(test_f1) + '\n'
                    )
                
                print("Finished:", model_name)
    
