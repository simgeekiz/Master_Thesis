
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Conv1D, \
                                    MaxPooling1D, Flatten, Reshape, \
                                    BatchNormalization, Dropout, add
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

from src.keras_bert import convert_text_to_examples, \
                           create_tokenizer_from_hub_module, \
                           convert_examples_to_features, \
                           initialize_vars, \
                           BertLayer

# Custom
from src.callbacks import PlotCurvesTF as PlotCurves
from src.eval_metrics import f1_micro, f1_macro
from src.load_data import load_data

sess = tf.compat.v1.Session()

################# MODELS #################

def build_model_0(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)

    pred = Dense(1, activation='sigmoid')(bert_output)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_1(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)

    x = Dense(256)(bert_output)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_2(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    def residual(x):
        x_res = x

        x = Dense(256)(x)
        x = Activation('relu')(x)

        x = add([x, x_res])
        return x

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)

    x = Dense(256)(bert_output)
    x = Activation('relu')(x)

    x = residual(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_3(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)
    bert_output = Reshape((768, 1))(bert_output)

    x = Conv1D(64, 5, padding='same')(bert_output)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(64, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_4(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)
    bert_output = Reshape((768, 1))(bert_output)

    x = Conv1D(128, 5, padding='same')(bert_output)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(128, 5, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_5(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)

    x = Dense(256)(bert_output)
    x = Activation('relu')(x)

    x = Dense(128)(bert_output)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_6(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    def residual(x):
        x_res = x

        x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(0.001))(x)
        x = Activation('relu')(x)

        x = add([x, x_res])
        return x

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)
    bert_output = Reshape((768, 1))(bert_output)

    x = Conv1D(64, 5, padding='same')(bert_output)
    x = Activation('relu')(x)

    x = residual(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_7(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)
    bert_output = Reshape((768, 1))(bert_output)

    x = Conv1D(64, 5, padding='same')(bert_output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(5)(x)

    x = Conv1D(64, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

def build_model_8(max_seq_length, n_fine_tune_layers=3,
                  bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    def residual(x):
        x_res = x

        x = Conv1D(64, 5, padding='same')(bert_output)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = add([x, x_res])
        return x

    bert_inputs = [Input(shape=(max_seq_length,), name="input_ids"),
                   Input(shape=(max_seq_length,), name="input_masks"),
                   Input(shape=(max_seq_length,), name="segment_ids")]

    bert_output = BertLayer(n_fine_tune_layers=n_fine_tune_layers,
                            pooling='mean', bert_path=bert_path)(bert_inputs)
    bert_output = Reshape((768, 1))(bert_output)

    x = Conv1D(64, 5, padding='same')(bert_output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual(x)
    x = MaxPooling1D(5)(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)

    pred = Dense(1, activation='sigmoid')(x)

    return Model(inputs=bert_inputs, outputs=pred)

################# UTILS #################


def get_input(data_, max_seq_length,
              bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    tokenizer = create_tokenizer_from_hub_module(bert_path)

    # !!! For BERT input, each sentence should be in an array
    X = np.array([[" ".join(sentence['sentence'].replace('\n', '').strip().split()[0:max_seq_length])]
                  for article in data_
                  for sentence in article['sentences']], dtype=object)

    y = [sentence['label']
         for article in data_
         for sentence in article['sentences']]

    examples_ = convert_text_to_examples(X, y)

    (input_ids, input_masks, segment_ids, labels_) = \
            convert_examples_to_features(tokenizer, examples_, max_seq_length=max_seq_length)

    return [input_ids, input_masks, segment_ids], labels_

def get_scores(model, data_, batch_size, max_seq_length, results_file=None, print_out=False):

    X, y_true = get_input(data_, max_seq_length)
    y = [y[0] for y in y_true]

    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = [round(y[0]) for y in y_preds]

    clsrpt = classification_report(y_true, y_preds)
    sfm = scikit_f1_score(y_true, y_preds, average='macro')

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

    batch_size = 32
    max_seq_length = 512
    if max_seq_length > 512:
        print('!!!!!!! WARNING: BERT does not accept lenght > 512. It is set to 512.')
        max_seq_length = 512

    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    build_models_functions = {
        'model_0_tunedlayers_3': build_model_0(max_seq_length, n_fine_tune_layers=3),
        'model_1_tunedlayers_3': build_model_1(max_seq_length, n_fine_tune_layers=3),
        'model_2_tunedlayers_3': build_model_2(max_seq_length, n_fine_tune_layers=3),
        'model_3_tunedlayers_3': build_model_3(max_seq_length, n_fine_tune_layers=3),
        'model_4_tunedlayers_3': build_model_4(max_seq_length, n_fine_tune_layers=3),
        'model_5_tunedlayers_3': build_model_5(max_seq_length, n_fine_tune_layers=3),
        'model_6_tunedlayers_3': build_model_6(max_seq_length, n_fine_tune_layers=3),
        'model_7_tunedlayers_3': build_model_7(max_seq_length, n_fine_tune_layers=3),
        'model_8_tunedlayers_3': build_model_8(max_seq_length, n_fine_tune_layers=3),
    }

    optimizer_configs = [
        {'name': 'adam',
         'lro': [0.001, 0.0001, 2e-5, 4e-5]},
        {'name': 'adamax',
         'lro': [0.001, 0.0001, 2e-5, 4e-5]},
        {'name': 'rmsprop',
         'lro': [0.001, 0.0001, 2e-5, 4e-5]},
    ]

    epoch_options = [1, 2, 3]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()

#     Limit for testing the pipeline
#     train_data = train_data[:1]
#     valid_data = valid_data[:1]
#     test_data = test_data[:1]

    X_tra, y_tra = get_input(train_data, max_seq_length)
    X_val, y_val = get_input(valid_data, max_seq_length)

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

                for epochs in epoch_options:

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

                    model_name = 'Optimized_RQ1_bert' + \
                                 '_' + fname + \
                                 '_' + optimizer_name +  \
                                 '_lr_' + str(lr) +  \
                                 '_epochs_' + str(epochs) + \
                                 '_loss_' + loss

                    model_main = './Model/' + model_name.split('model')[0] + 'model/'
                    model_dir = os.path.join(model_main, model_name)
                    score_file = os.path.join(model_main, 'model_performances.csv')
                    results_file = os.path.join(model_dir, 'model_results_file.txt')

                    print("Fitting the model...")

                    # Instantiate variables
                    initialize_vars(sess)

                    model.fit(X_tra, y_tra,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(X_val, y_val),
                              callbacks=[
                                  PlotCurves(model_name=model_name,
                                             model_dir=model_dir,
                                             plt_show=False,
                                             jnote=False)
                              ])

                    print("Evaluating the model...")

                    with open(results_file, 'w') as f:
                        f.write('\n---------------- Validation ----------------\n')

                    val_f1 = get_scores(model, valid_data, batch_size, max_seq_length, results_file)

                    with open(results_file, 'a') as f:
                        f.write('\n---------------- Test ----------------\n')

                    test_f1 = get_scores(model, test_data, batch_size, max_seq_length, results_file)

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
