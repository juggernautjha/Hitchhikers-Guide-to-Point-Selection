
"""Contains an interface for playing and evaluating Hangman. I am shameless and will therefore copy trexquant's code and make necessary improvements. 
"""


import json
import requests
import random
import string
import secrets
import time
import re
import collections
import typing
import numpy as np
from copy import deepcopy
try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode
import tensorflow as tf
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from .generator import IDX_TO_CHR, CHR_TO_IDX, I_T_C, C_T_I, pad_and_encode

class Hangman(object):
    def __init__(
            self, dictionary : str = "data/cleaned_test.txt", lives : int = 6, mode : int = 1
    ):
        self.wordlist = [i.strip() for i in open(dictionary).readlines()]
        self.lives = lives
        self.idx = 0
        self.mode = mode

    def start_game(self) -> typing.Tuple:
        """Returns self.state and self.status
        """
        self.word = self.wordlist[self.idx] if self.mode == 0 else random.choice(self.wordlist)
        self.letters = set([i for i in self.word])
        self.state = '_'*len(self.word)
        self.rem_lives = self.lives
        self.status = 0 #! 1 if win, -1 if loss. 
        self.used = set([])
        self.history = [(self.state, np.zeros(26), self.word)]
        self.idx += 1
        return self.state, self.status, self.rem_lives
    
    def update(self, char : str):
        if self.status != 0:
            return RuntimeError
        # print(f"Before Updating: \              {self.state} {self.word} {self.used}")
        # print(f"You guessed : {char}")
        
        if char in self.letters and char not in self.used:
            tmp = [self.state[i] if self.word[i] != char else char for i in range(len(self.state))]
            self.state = ''.join(tmp)
        else:
            self.rem_lives -= 1

        tmp = deepcopy(self.history[-1][-2])
        tmp[ord(char) - ord('a')] = 1
        

        # print(f"After: \              {self.state} {self.word} {self.used}")
        
        self.history.append(
            (self.state, tmp, self.word)
        )

        if self.state == self.word:
            self.status = 1

        if self.rem_lives <= 0:
            self.status = -1


        
        return self.state, self.status, self.rem_lives
    

    


class HangmanAPIMod(object):
    """No ML Involved"""
    def __init__(self):
        self.guessed_letters = []
        
        full_dictionary_location = "data/cleaned_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word.replace("_",".")
        # print(clean_word)
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        

        # print(new_dictionary)
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
                    
        # print(guess_letter)
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, gameobj = Hangman("data/cleaned_test.txt"),practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        
        state, code = gameobj.start_game() #! This should reset it. 
        # print(f"THE WORD IS {gameobj.word}")

        while code == 0:
            # print(f"state : {state} status : {'ongoing' if code == 0 else 'over'}")
            my_guess = self.guess(state)
            # print(f"guess : {my_guess}")
            state, code, lives = gameobj.update(my_guess)
            self.guessed_letters.append(my_guess)
        

        return code, gameobj.word, 6
    

    def generate_data(self, environ : Hangman):
        self.start_game(environ)
        state = environ.history
        return state

        
    

class HangmanML(HangmanAPIMod):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def guess(self, word):
        """Overwriting hehe"""
        encoded_word = pad_and_encode(word)
        encoded_word = np.array([encoded_word])
        result = self.model(encoded_word)
        guess_idx = tf.argsort(result[0], direction='DESCENDING')
        for i in guess_idx:
            c = IDX_TO_CHR[i.numpy()]
            if c not in self.guessed_letters:
                return c
        return IDX_TO_CHR[guess_idx[0].numpy()]



class HangmanStatefulML(HangmanAPIMod):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.used = np.array([0]*26)

    def guess(self, word):
        """We be overwriting"""
        encoded_word = pad_and_encode(word)
        encoded_word = np.array([encoded_word])
        used_letters = np.array([self.used])
        result = self.model([encoded_word, used_letters])
        guess_idx = tf.argsort(result[0], direction='DESCENDING')
        for i in guess_idx:
            c = IDX_TO_CHR[i.numpy()]
            if c not in self.guessed_letters:
                return c
        return IDX_TO_CHR[guess_idx[0].numpy()]
    

    def start_game(self, gameobj = Hangman("data/cleaned_test.txt"),practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        
        state, code = gameobj.start_game() #! This should reset it. 
        # print(f"THE WORD IS {gameobj.word}")

        while code == 0:
            # print(f"state : {state} status : {'ongoing' if code == 0 else 'over'}")
            my_guess = self.guess(state)
            # print(f"guess : {my_guess}")
            state, code = gameobj.update(my_guess)
            self.guessed_letters.append(my_guess)
            self.used[ord(my_guess) - ord('a')] = 1
        

        return code, gameobj.word  







    
