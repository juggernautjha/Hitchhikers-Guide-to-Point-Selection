import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from tensorflow.keras import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, Conv1D, Add, Multiply, Lambda, Conv2DTranspose, Concatenate, UpSampling2D, Reshape, Dot, Permute, RepeatVector, Embedding
import tensorflow as tf
import typing
from keras.utils import Sequence
import numpy as np
from generators.text_generator import C_T_I, I_T_C, CHR_TO_IDX, IDX_TO_CHR
from tqdm.notebook import tqdm

#! EVAL SCRIPT
def eval_pretty(generator, model, DEBUG = False):
    total = 0
    correct = 0
    pbar = tqdm(range(len(generator)))
    for idx in pbar:
        inp, out = generator.__getitem__(idx)
        pred = model(inp)
        maxs = tf.argmax(pred, 1)
        for i, mx in enumerate(maxs):
            nz = tf.reshape(tf.where(out[i] > 0), (1,-1))
            kz = [IDX_TO_CHR[j.numpy()] for j in nz[0]]
            predicted = IDX_TO_CHR[mx.numpy()]
            score = 0
            
            if predicted in kz:
                correct += 1
                score = 1
            if DEBUG: print(f"Predicted : {predicted} Expected : {kz} Score : {score}")
            total += 1
        pbar.set_description(f"{correct/total}")

#! ENCODING MODELS
def LSTM_ENC(length : int, embed_dim : int = 5, lstm_dims : typing.List[int] = [64, 32], dense_dims : typing.List[int] = [10]):
    """encodes the current game state

    Args:
        length (int): max_length. Need to make it a parameter for rigourous testing.
        embed_dim (int, optional): embedding vector size. Defaults to 5.

    Returns:
        Model: a model that takes in game state and returns a vector. 
    """
    input_ = Input(shape=(length,))
    x = tf.keras.layers.Embedding(input_dim = 31, output_dim = embed_dim, mask_zero=True)(input_)
    #! Getting LSTM Embedding


    for i, dim in enumerate(lstm_dims[:-1]):
        x  = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dim,  return_sequences=True), name=f"lstm_encoding_{i}")(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dims[-1]), name=f"lstm_encoding_{len(lstm_dims)}")(x)  
    for i, dim in enumerate(dense_dims): 
        x = tf.keras.layers.Dense(dim, name=f"bottle_neck_enc{i}")(x)
    
    output = tf.keras.layers.Dense(26, activation="softmax")(x)
    return Model( input_,  output)


def encoder_stub(length : int, embed_dim : int = 5, lstm_dims : typing.List[int] = [64, 32], dense_dims : typing.List[int] = [10]):
    """encodes the current game state

    Args:
        length (int): max_length. Need to make it a parameter for rigourous testing.
        embed_dim (int, optional): embedding vector size. Defaults to 5.

    Returns:
        Model: a model that takes in game state and returns a vector. 
    """
    input_ = Input(shape=(length,))
    x = tf.keras.layers.Embedding(input_dim = 31, output_dim = embed_dim, mask_zero=True)(input_)
    #! Getting LSTM Embedding


    for i, dim in enumerate(lstm_dims[:-1]):
        x  = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dim,  return_sequences=True), name=f"lstm_encoding_{i}")(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dims[-1]), name=f"lstm_encoding_{len(lstm_dims)}")(x)  
    for i, dim in enumerate(dense_dims): 
        x = tf.keras.layers.Dense(dim, name=f"bottle_neck_enc{i}")(x)
    
    output = x
    return Model( input_,  output)

def GRU_ENC(length : int, embed_dim : int = 5, lstm_dims : typing.List[int] = [128,128, 64], dense_dims : typing.List[int] = [60,20,6]):
    input_ = Input(shape=(length,))
    x = tf.keras.layers.Embedding(input_dim = 31, output_dim = 20, mask_zero=True)(input_)
    #! Getting LSTM Embedding


    for i, dim in enumerate(lstm_dims[:-1]):
        x  = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(dim,  return_sequences=True), name=f"lstm_encoding_{i}")(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_dims[-1]), name=f"lstm_encoding_{len(lstm_dims)}")(x)
    for i, dim in enumerate(dense_dims):
        x = tf.keras.layers.Dense(dim, name=f"bottle_neck_enc{i}")(x)

    output = tf.keras.layers.Dense(26, activation="softmax")(x)
    return Model( input_,  output)


def baseline_dense_concat(length : int, embed_dim : int = 5, dense_dims : typing.List[int] = [128, 64, 32]):
    input_ = Input(shape=(length,))
    x = tf.keras.layers.Embedding(input_dim = 31, output_dim = 20, mask_zero=True)(input_)
    x = tf.keras.layers.Flatten()(x)
    for i, dim in enumerate(dense_dims):
        x = tf.keras.layers.Dense(dim, name=f"bottle_neck_enc{i}")(x)
    output = tf.keras.layers.Dense(26, activation="softmax")(x)
    return Model( input_,  output)




