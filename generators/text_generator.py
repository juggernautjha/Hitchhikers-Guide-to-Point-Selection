
# Write a datagenerator using TF Dataset
# Maybe add methods to oversample small words for more stringent training??
# I want this file to be model agnostic, give it the filename, and it generates a dataloader. 

import tensorflow as tf
import numpy as np
try :
    from keras.utils import Sequence #   sequence =  keras.utils.Sequence
except:
    from keras.utils.all_utils import Sequence

import tensorflow.keras as K
import os
import sys
import typing
import random

import pickle

#! 
IDX_TO_CHR = {
    i : chr(ord('a') 
            + i) for i in range(0, 27)
}

CHR_TO_IDX = {
    j : i for i,j in IDX_TO_CHR.items()
}


#!
I_T_C = {
    i+1 : chr(ord('a') 
            + i) for i in range(0, 27)
}
I_T_C[27] = '_'

C_T_I ={
    j : i for i, j in I_T_C.items()
}

def mask(word : str, msk ):
    """Ensure consistent masking"""
    tmp_wrd =  "".join([word[i] if msk[i] == 0 else '_' for i in range(len(word))])
    diff = [word[i] for i in range(len(word)) if msk[i] == 1]
    final_wrd = "".join([tmp_wrd[i] if tmp_wrd[i] not in diff else '_' for i in range(len(word))])
    return final_wrd

def pad_and_encode(word : str, length : int = 31):
    wrd_lst = [C_T_I[i] for i in word]
    lng = length - len(word) if length >= len(word) else 0 
    wrd_lst = wrd_lst + lng*[0]
    return wrd_lst

def get_probability_vector(input : str, output : str):
        """Get the probability vector.

        Args:
            input (str) 
            output (str)

        Returns:
            tf.tensor -> expected probability vector.
        """

        diff = [output[i] for i in range(len(output)) if output[i] != input[i]]
        lst = [0]*26
        for d in diff:
            # print(CHR_TO_IDX[d])
            lst[CHR_TO_IDX[d]] += 1

        if len(diff) != 0:
            lst = np.array(lst)/len(diff)
        else:
            lst = np.array(lst)
        return list(lst)  
    
class pretraining_generator(Sequence):
    """Generator for generating pretraining data.

    Args:
        filename : str, filename to generate examples from.
        samples_per_word : int, number of examples to generate per word.
        batch_size : int, batch size to generate.

    Also, do not kill me, I am going to inherit from this to create the games generator. Just need to run quick tests :).
    """

    def __init__(self, filename : str, samples_per_word : int, batch_size : int = 16, pad : bool = False, length : int = 31, shuffle : bool = True):
        self.words = [i.strip() for i in open(filename).readlines()]

        assert batch_size > samples_per_word, "Not really a problem, but can you please use a larger batch size"
        self.samples_per_word = samples_per_word
        self.batch_size = batch_size
        self.step = self.batch_size//self.samples_per_word
        self.shuffle = shuffle
        self.pad = pad
        self.length = length
        random.shuffle(self.words)

    def __len__(self):
        return (len(self.words)*self.samples_per_word)//self.batch_size
    
    def on_epoch_end(self):
        random.shuffle(self.words)

    def generate_training_data(self, word : str, num_samples : int):
        inp = []
        out = []
        lnwrd = len(word)
        masks = np.random.randint(2, size=(num_samples, lnwrd))
        for i in range(num_samples):
            masked = mask(word, masks[i])
            if masked != word:
                inp.append(masked)
        for wrd in inp:
            out.append(get_probability_vector(wrd, word))
        return inp, out  

    def __getitem__(self, idx : int):
        words = self.words[idx*self.step : min(len(self.words), (idx+1)*self.step)]
        inputs = []
        outputs = []
        for word in words:
            inp, out = self.generate_training_data(word, self.samples_per_word)
            inputs += inp
            outputs += out
        c = list(zip(inputs, outputs))
        random.shuffle(c)
        inputs, outputs = zip(*c)
        if self.pad:
            inputs = [pad_and_encode(i, self.length) for i in inputs]
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        return inputs, outputs


class state_generator(Sequence):
    """Generates (state, used) pairs for better training"""
    def __init__(self, filename: str, batch_size: int = 16, length: int = 31, shuffle: bool = True, rtrn_used : bool = True):
        self.filename = filename
        self.triples = pickle.load(open(filename, "rb"))
        self.triples = [i for i in self.triples if i[0] != i[-1]]
        if shuffle:
            random.shuffle(self.triples)
        self.batch_size = batch_size
        self.length = length
        self.used_return = rtrn_used
        self.on_epoch_end()
    
    def __len__(self):
        return (len(self.triples))//self.batch_size
    
    def on_epoch_end(self):
        random.shuffle(self.triples) 

    def __getitem__(self, idx : int):
        triples = self.triples[idx*self.batch_size : min(len(self.triples), (idx+1)*self.batch_size)]
        states = []
        used = []
        probabs = []
        words = []

        for triple in triples:
            state, usd, word = triple
            prob_vec = get_probability_vector(state, word) 
            # print(sum(prob_vec))
            states.append(state)
            used.append(usd)
            probabs.append(prob_vec)
            words.append(word)

        if 1==1:
            states = [pad_and_encode(i, self.length) for i in states]
        
        states = np.array(states)
        used = np.array(used)
        probabs = np.array(probabs)
        words = np.array(words)
        if self.used_return:
            return [states, used], probabs
        else: return states, probabs



class rho_generator_audio(pretraining_generator):
    def __init__(self, filename : str, samples_per_word : int, batch_size : int = 16, pad : bool = False, length : int = 31, shuffle : bool = True, selector = None, irred_model : str = '', target_model : str = '', target_model_path : str = '', epoch_cutoff : int = 3, minibatch_size : float = 0.6, loss : tf.keras.losses.Loss = tf.keras.losses.categorical_crossentropy):
        super().__init__(filename, samples_per_word, batch_size, pad ,length, shuffle)
        self.selector = selector
        self.irred_model = irred_model
        self.target_model_path = target_model_path
        self.target_model = target_model
        self.epoch_cutoff = epoch_cutoff
        self.minibatch_size = minibatch_size
        self.loss = loss

        self.select_func = self.selector(self.irred_model, self.target_model, self.minibatch_size);
        self.target_model = None

        self.cache = None

        #start rho_selection after epoch_cutoff
        #! selector takes in pretrained model, target model, and returns a list of indices


    def on_epoch_end(self):
        # def on_epoch_end(self):
        random.shuffle(self.words)
        try:
            self.target_model = tf.keras.models.load_model(self.target_model_path)
            self.target_model.compile(optimizer = 'adam', loss = self.loss, metrics = ['accuracy'])

            self.select_func = self.selector(self.irred_model, self.target_model, self.minibatch_size)
        except Exception as e:
            pass
        print(self.cache)
        self.epoch_cutoff = self.epoch_cutoff - 1 if self.epoch_cutoff > 0 else 0
        print("\n")

    def __getitem__(self, index):
        a, b = super().__getitem__(index)
        
        if self.selector is not None and self.epoch_cutoff <= 0:
            if self.select_func is None:
                self.select_func = self.selector(self.irred_model, self.target_model, self.minibatch_size)
            indices = self.select_func(a, b)
            a = a[indices]
            b = b[indices]
        self.cache = (a.shape, b.shape)
        return a, b