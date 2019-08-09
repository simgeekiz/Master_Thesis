import tensorflow as tf
from sklearn.metrics import classification_report#, accuracy_score
# from sklearn import metrics

def f1_score(y_true, y_pred):
#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

def acc_seq(y_true, y_pred):
    
    y_true = flatten_y(y_true)
    y_pred = flatten_y(y_pred)
    
    return accuracy_score(y_true, y_pred)