
import numpy as np
import tensorflow as tf

def f1_macro(y_true, y_pred):
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")

    # tp = [10, 1]: each column is tp of a class
    tp = tf.reduce_sum(y_true*y_pred, 0)
    fp = tf.reduce_sum((1-y_true)*y_pred, 0)
    fn = tf.reduce_sum(y_true*(1-y_pred), 0)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    f1 = 2 * p * r / (p + r)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return tf.reduce_mean(f1)

def f1_micro(y_true, y_pred):
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32")
    
    tp = tf.reduce_sum(y_true*y_pred, axis=1)
    tp_fn = tf.reduce_sum(y_true, axis=1)
    tp_fp = tf.reduce_sum(y_pred, axis=1)
        
    p = tp / tp_fp
    r = tp / tp_fn
    
    f1 = 2 * p * r / (p + r)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return tf.reduce_mean(f1)