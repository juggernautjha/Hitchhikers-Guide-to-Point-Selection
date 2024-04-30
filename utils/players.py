
#! All players inherit from the base Hangman API.

import json
import requests
import random
import string
import secrets
import time
import re
import collections
from collections import Counter
import operator
try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning
from tqdm.notebook import tqdm
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import pickle
from . import game
import numpy as np
from . import generator
import tensorflow as tf


def vowel_frequencies(words):
    vowel_counts = {}
    vowels = set('aeiou')

    for word in tqdm(words):
        length = len(word)
        if length not in vowel_counts:
            vowel_counts[length] = Counter()

        for char in word:
            if char in vowels:
                vowel_counts[length][char] += 1

    # Sort vowels by frequency for each word length
    for length in vowel_counts:
        vowel_counts[length] = sorted(vowel_counts[length].items(), key=operator.itemgetter(1), reverse=True)

    return vowel_counts


def alephfreq(words):
    vowel_counts = {}
    alephs = [chr(i) for i in range(ord('a'), ord('z')+1)]
    vowels = set(alephs)

    for word in tqdm(words):
        length = len(word)
        if length not in vowel_counts:
            vowel_counts[length] = Counter()

        for char in word:
            if char in vowels:
                vowel_counts[length][char] += 1

    # Sort vowels by frequency for each word length
    for length in vowel_counts:
        vowel_counts[length] = sorted(vowel_counts[length].items(), key=operator.itemgetter(1), reverse=True)

    return vowel_counts

#! BASE API
class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []
        
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
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
        
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)

class HangmanModded(HangmanAPI):
    """
    Base API Modified to use custom guess function.
    """
    def __init__(self, model, dictionary : str = "data/words_250000_train.txt", 
                  access_token=None, session=None, timeout=2000, 
                  DEBUG = True, WEB=True, mode='vowel', threshold=5):
        self.model = model
        self.DEBUG = DEBUG
        self.guessed_letters = []
        self.dictionary = [i.strip() for i in open(dictionary).readlines()]
        self.is_word = set(self.dictionary)
        self.ngrams = self.generate_ngrams()
        if mode == 'vowels':
          self.vowels = vowel_frequencies(self.dictionary)
        else:
          self.vowels = alephfreq(self.dictionary)
        self.thresh = threshold #Initial guesses using probab
        self.access_token = access_token
        self.hangman_url='https://sg.trexsim.com/trexsim/hangman'
        self.session = session or requests.Session()
        self.timeout = timeout


    
    def get_kgram(self, word: str, k = int):
        if k > word.__len__():
            return []
        kgrams = []
        for i in range(0, len(word) - k+ 1):
            tmp = word[i:i+k]
            if tmp in self.is_word:
                kgrams.append((tmp, 1)) #kgram is also a complete word.
                if self.DEBUG: print(f"{tmp} {word} {k}")
            else:
                kgrams.append((tmp, 0))
        return kgrams

    def generate_ngrams(self, mini = 2, maxi = 15):
        try:
            ngrams = pickle.load(open("data/ngrams_pickled.pkl", "rb"))
        except:
            ngrams = {
                i : [] for i in range(mini, maxi+1)
            }
            print("Loading NGrams -> This is a one-time process :)")
            for i in tqdm(range(mini, maxi+1)):
                for word in tqdm(self.dictionary):
                    ngrams[i] += self.get_kgram(word, i)
            ngrams = {
                i : Counter(ngrams[i]) for i in ngrams
            }
            pickle.dump(ngrams, open("data/ngrams_pickled.pkl", "wb"))
        
        return ngrams
    

    def start_game_web(self, practice=True, verbose=False):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.dictionary
        # self.ngrams 


        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            
            tries_remains = response.get('tries_remains')
            if verbose:
                
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                word = word[::2]
                # print(word)
                # get guessed letter from user code
                guess_letter = self.guess(word, tries_remains, self.thresh)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
    


    def classic_guess(self, word):
      if self.thresh > 6:
        print("HEHEHE")
      my_len = len(word)
      vowel_count = self.vowels[my_len]
      for i, _ in vowel_count:
          if i not in self.guessed_letters:
              return i
      else:
          return self.mlguess(word)

    def ml_guess(self, word):
      encoded_word = generator.pad_and_encode(word)
      encoded_word = np.array([encoded_word])
      result = self.model(encoded_word)
      guess_idx = tf.argsort(result[0], direction='DESCENDING')
      for i in guess_idx:
          c = generator.IDX_TO_CHR[i.numpy()]
          if c not in self.guessed_letters:
              return c
      return generator.IDX_TO_CHR[guess_idx[0].numpy()]



    def guess(self, word, lives, thresh = 5):
        if lives < thresh:
          return self.ml_guess(word)
        else:
          return self.classic_guess(word)

    def start_game_local(self, gameobj = game.Hangman()):
        self.guessed_letters = []
        self.current_dictionary = self.dictionary
        state, code, lives_rem = gameobj.start_game() #! This should reset it. 
        # print(f"THE WORD IS {gameobj.word}")

        while code == 0:
            # print(f"state : {state} status : {'ongoing' if code == 0 else 'over'}")
            my_guess = self.guess(state, lives_rem, self.thresh)
            # print(f"guess : {my_guess}")
            state, code, lives_rem = gameobj.update(my_guess)
            self.guessed_letters.append(my_guess)
        

        return code, gameobj.word


    


    


    

    

        









