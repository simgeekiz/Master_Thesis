import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Bidirectional, \
                                    Dropout, LSTM, add
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax
from tensorflow.keras.regularizers import l2

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as scikit_f1_score

from src.keras_bert import convert_single_example, \
                           convert_text_to_examples, \
                           create_tokenizer_from_hub_module, \
                           convert_examples_to_features, \
                           InputExample, \
                           initialize_vars

# Custom
from src.callbacks import PlotCurvesTF as PlotCurves
from src.eval_metrics import f1_macro, f1_micro
from src.load_data import load_data

sess = tf.compat.v1.Session()

with tf.device("gpu:0"):
    print("GPU enabled")

os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

bert = hub.Module(bert_path, trainable=True)

ww = 1
max_seq_length = 512
if max_seq_length > 512:
    print('!!!!!!! WARNING: BERT does not accept length > 512. It is set to 512.')
    max_seq_length = 512

def BERTEmbeddingStack(x):
    embeds = []
    for art in tf.unstack(tf.reshape(x, (batch_size, 3, 2*ww+1, max_seq_length))):
        art = tf.cast(art, dtype="int32")
        # Below does not change the shape of segment_ids etc.
        # Only puts them into a dictionary
        bert_inputs = dict(
            input_ids=art[0],
            input_mask=art[1],
            segment_ids=art[2]
        )
        # Pooling
        result = bert(bert_inputs, signature="tokens", as_dict=True)["sequence_output"]
        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
        masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                             tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
        input_mask = tf.cast(art[1], tf.float32)
        pooled = masked_reduce_mean(result, input_mask)
        embeds.append(pooled)
    # print(tf.stack(embeds, 0))
    return tf.stack(embeds, 0)

################# MODELS #################


def build_model_0(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(bert_output)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_1(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(bert_output)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_2(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(bert_output)
    x = Dropout(0.4)(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_3(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(bert_output)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_4(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(bert_output)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_5(ww, max_seq_length):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(bert_output)
    x = Dropout(0.4)(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_6(ww):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(bert_output)
    x = Dropout(0.4)(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_7(ww):

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(bert_output)
    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_8(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(bert_output)

    x = residual(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_9(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(bert_output)

    x = residual(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_10(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(bert_output)

    x = residual(x)

    x = Dropout(0.4)(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_11(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=128, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(bert_output)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True))(x)

    x = residual(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_12(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=256, return_sequences=True))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(bert_output)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=256, return_sequences=True))(x)
    x = residual(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)


def build_model_13(ww, max_seq_length):

    def residual(x):
        x_res = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x_res, x])
        return x

    inp_size = 2 * ww + 1
    input_text = Input(shape=(3, inp_size, max_seq_length))

    bert_output = Lambda(BERTEmbeddingStack, output_shape=(None, None, inp_size, 768))(input_text)

    x = Dense(256, kernel_regularizer=l2(0.001))(bert_output)
    x = Activation('relu')(x)

    x = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)

    x = residual(x)

    pred = LSTM(1, activation='sigmoid')(x)

    return Model(inputs=[input_text], outputs=pred)

################# UTILS #################


# def get_padding_sentence(max_seq_length, tokenizer, padding_text='ENDPAD'):
#     """Deprecated"""
#
#     example_sent = InputExample(guid=None, text_a=" ".join(padding_text), text_b=None, label=0)
#
#     (input_ids, input_mask, segment_ids, label) = \
#         convert_single_example(tokenizer, example_sent, max_seq_length=max_seq_length)
#
#     return {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label": 0}


def get_input(data_, ww, max_seq_length):

    tokenizer = create_tokenizer_from_hub_module(bert_path)

    padding_sent = {"input_ids": [0]*max_seq_length, "input_mask": [0]*max_seq_length,
                    "segment_ids": [0]*max_seq_length, "label": 0}

    X = []
    y = []
    for article in data_:

        X_art = np.array([[" ".join(sentence['sentence'].replace('\n', ' ').strip().split()[0:max_seq_length])]
                          for sentence in article['sentences']], dtype=object)

        y_art = [sentence['label'] for sentence in article['sentences']]

        examples_ = convert_text_to_examples(X_art, y_art)

        (input_ids, input_masks, segment_ids, labels_) = \
            convert_examples_to_features(tokenizer, examples_, max_seq_length=max_seq_length)

        for si, _ in enumerate(article['sentences']):

            input_ids_seq = []
            input_mask_seq = []
            segment_ids_seq = []
            y_seq = []

            # Prev
            for i in reversed(range(ww)):

                if si - i - 1 >= 0:
                    sent_obj_prev = {"input_ids": input_ids[si - i - 1],
                                     "input_mask": input_masks[si - i - 1],
                                     "segment_ids": segment_ids[si - i - 1]}
                else:
                    sent_obj_prev = padding_sent

                input_ids_seq.append(sent_obj_prev['input_ids'])
                input_mask_seq.append(sent_obj_prev['input_mask'])
                segment_ids_seq.append(sent_obj_prev['segment_ids'])

            # Curr
            sent_obj = {"input_ids": input_ids[si],
                        "input_mask": input_masks[si],
                        "segment_ids": segment_ids[si]}

            input_ids_seq.append(sent_obj['input_ids'])
            input_mask_seq.append(sent_obj['input_mask'])
            segment_ids_seq.append(sent_obj['segment_ids'])
            y_seq.append(labels_[si][0])

            # Next
            for i in range(ww):

                if si + i + 1 < len(article['sentences']):
                    sent_obj_next = {"input_ids": input_ids[si + i + 1],
                                     "input_mask": input_masks[si + i + 1],
                                     "segment_ids": segment_ids[si + i + 1]}
                else:
                    sent_obj_next = padding_sent

                input_ids_seq.append(sent_obj_next['input_ids'])
                input_mask_seq.append(sent_obj_next['input_mask'])
                segment_ids_seq.append(sent_obj_next['segment_ids'])

            X_seq = (np.array(input_ids_seq),
                     np.array(input_mask_seq),
                     np.array(segment_ids_seq))

            X.append(X_seq)
            y.append(y_seq)

    return np.array(X), np.array(y)


def get_scores(model, batch_size, ww, max_seq_length, data_=None, X=None, y_true=None, results_file=None, print_out=False):

    if data_:
        X, y_true = get_input(data_, ww, max_seq_length, batch_size, limit=None)
    y_true = [y[0] for y in y_true]

    y_preds = model.predict(X, batch_size=batch_size)
    y_preds = [0 if y[0] < 0.5 else 1 for y in y_preds]

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

    batch_size = 2

    model_no = sys.argv[1]
    optimizer_name = sys.argv[2]
    lr = float(sys.argv[3])

    epochs = 1

    loss = 'binary_crossentropy'
    metrics = ['acc', f1_macro, f1_micro]

    #### LOAD DATA ####

    train_data, valid_data, test_data, _ = load_data()
    for a, article in enumerate(valid_data):
        for s, sent in enumerate(article['sentences']):
            lent = len(sent['sentence'].split())
            if lent > 90:
                del valid_data[a]['sentences'][s]

    # Limit for testing the pipeline
    # train_data = train_data[:1]
    # valid_data = valid_data[:1]
    # test_data = test_data[:1]

    X_tra, y_tra = get_input(train_data, ww, max_seq_length, batch_size, limit=None)
    X_val, y_val = get_input(valid_data, ww, max_seq_length, batch_size, limit=None)

    del train_data

    #### CONFIGURE MODEL ####

    model = getattr(sys.modules[__name__], 'build_model_' + model_no)(ww, max_seq_length=max_seq_length)

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

    model_name = 'Optimized_RQ2_bert_many_to_one' + \
                 '_model_' + model_no + \
                 '_ww_' + str(ww) + \
                 '_' + optimizer_name + \
                 '_lr_' + str(lr) + \
                 '_epochs_' + str(epochs) + \
                 '_loss_' + loss + \
                 '_sigmoid'

    model_main = './Model/' + model_name.split('model')[0] + 'model/'
    model_dir = os.path.join(model_main, model_name)
    score_file = os.path.join(model_main, 'model_performances.csv')
    results_file = os.path.join(model_dir, 'model_results_file.txt')

    print("\n----------------------------------\nFitting the model: ", model_name)

    # Instantiate variables
    initialize_vars(sess)

    model.fit(X_tra, y_tra,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_val, y_val),
              callbacks=[
                  PlotCurves(model_name=model_name, model_dir=model_dir,
                             plt_show=False, jnote=False, save_best=False),
              ])

    print("Evaluating the model...")

    with open(results_file, 'w') as f:
        f.write('\n---------------- Validation ----------------\n')

    val_f1 = get_scores(model, batch_size, ww, max_seq_length, X=X_val, y_true=y_val, results_file=results_file)

    with open(results_file, 'a') as f:
        f.write('\n---------------- Test ----------------\n')

    test_f1 = get_scores(model, batch_size, ww, max_seq_length, data_=test_data, results_file=results_file)

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
