
#! Implementing loss based and other selection strats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import random, glob, os
import numpy as np

from pydub import AudioSegment
from pydub import effects
from utils.refactored_common import *
# from utils.refactored_common import unision_shuffled_copies
from tqdm.notebook import tqdm
try :
    from keras.utils import Sequence #   sequence =  keras.utils.Sequence
except:
    from keras.utils.all_utils import Sequence

import numbers
import warnings

def calculate_irreducible_loss(irred_model : tf.keras.Model, x : np.ndarray, y : np.ndarray) -> tf.Tensor:
    """Calculates the irreducible loss of the given dataset. 

    Args:
        irred_model (tf.keras.Model): model pretrained on holdout dataset
        data (Sequence): Generator yielding batches of data
    """
    irreducible_loss = irred_model.evaluate(x, y)
    return irreducible_loss

class irreducible_loss_selector:
    def __init__(self, target_model : tf.keras.Model, irred_model : tf.keras.Model, minibatch_size : float, loss : tf.keras.losses.Loss = tf.keras.losses.categorical_crossentropy):
        self.target_model = target_model
        self.irred_model = irred_model
        self.minibatch_size = minibatch_size
        self.loss = loss
        
    def __call__(self,
                 x : np.ndarray,
                 y : np.ndarray) ->tf.Tensor:
        """Given a batch, selects the subset of the batch that maximizes the rholoss

        Args:
            target_model (tf.keras.Model): model to train
            irred_model (tf.keras.Model): model pretrained on holdout dataset
            batch (tf.Tensor): batch of data 
            minibatch_size (float): proportion the minibatch needed to be selected

        Returns:
            tf.Tensor: indices to train on
        """
        irred_points = self.irred_model(x)
        red_points = self.irred_model(x)
        reducible_loss = self.loss(y, red_points)
        irreducible_loss = self.loss(y, irred_points)
        rholoss = reducible_loss - irreducible_loss

        selected_indices = tf.argsort(rholoss, direction='DESCENDING')[:int(self.minibatch_size * len(rholoss))]
        
        return selected_indices
