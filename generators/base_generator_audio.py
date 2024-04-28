import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import random, glob, os
import numpy as np

from pydub import AudioSegment
from pydub import effects
# from utils.refactored_common import *
# from utils.refactored_common import unision_shuffled_copies
from tqdm.notebook import tqdm
import pydub
import librosa
try :
    from keras.utils import Sequence #   sequence =  keras.utils.Sequence
except:
    from keras.utils.all_utils import Sequence


# import tensorflow_io as tfio

import soundfile as sf
import audioflux
from scipy import signal

def gen_mel_feature (param:dict, data : pydub.AudioSegment, Fs : int, n_fft : int, hop_length : int, win_length : int, n_mels : int, log=True, window='hann', enable_librosa=False, gen_spec=False, gen_stft=False):
    global kym_model
    n = 12 if n_fft == 4096 else 11
    if n_fft == 8192 :n = 13
    window_value = param['feature']['window_types'][window]
    log_eps = 1e-6

    # 2nd normalize -> pcen normalization 
    if param['feature']['pcen_norm']:
        data = librosa.pcen(data, sr=Fs, hop_length=hop_length)

    # for librosa
    if window == 'gauss':
        window = (window, 7) # gauss window-> (window_name, std-sigmavalue) -> ('gauss', 7) | 7 is mentioned in scipy website example
    elif window == "kaiser":
        window = (window, 5) # kaiser -> (window, beta) --> beta -5, in audioflux for kaiser window 5 is the default value for beta
    else: # for other windows in config name only enough
        pass 

    if gen_spec:
        spec = audioflux.BFT(num=n_mels, radix2_exp=n, samplate=Fs, slide_length=hop_length)
        t = spec.bft(data,result_type=1) 
    
    elif gen_stft: 
        t = librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
        t = np.abs(t)
    

    else :
        if enable_librosa == True:
            t = librosa.feature.melspectrogram(y=data, sr=Fs, n_fft=n_fft, hop_length=hop_length, 
                                                    win_length=n_fft, window=window, center=False, n_mels=n_mels)
        else: 
            s = audioflux.MelSpectrogram(num=n_mels, samplate=Fs, radix2_exp=n, slide_length=hop_length)
            t = s.spectrogram(data)
  
    # 3rd normalize options -> mel normalization
        
    if log:
        t = np.log( np.maximum(log_eps, t ))

    if param['feature']['mel_norm']:
        t = (t-np.mean(t))/(np.std(t)+log_eps)

    t = t.T
    return t






class BaseClassificationGenerator(Sequence):
    """Base class for audio data generators. I hope to just use this. 
    Args:
        base_dir : str : The base directory where the audio files are stored
        batch_size : int : The batch size for the generator
        shuffle : bool : Whether to shuffle the data or not
        gentype: str : The type of generator to create. Options are 'train', 'test', 'holdout'

        Note: the base_dir should have the following structure:
        base_dir/gen_type/class_name/file_name.ext
    
    """
    
    #! We will use the default options for spectrogram
    clip = 700
    def __init__(self, param, base_dir : str, batch_size : int = 16, shuffle : bool = True, gentype : str = 'train', return_spec : bool = False, return_fft : bool = False, ext : str = 'flac'):
        self.ext = ext
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gentype = gentype
        self.classes = self.get_classes()
        self.class_dict = {c:i for i,c in enumerate(self.classes)}
        self.files = self.get_files()
        self.param = param
        self.n_fft = param['feature']['n_fft'] if param['feature']['Fs'] == 48000 else 2048 # 64 ms
        self.hop_length = param['feature']['hop_length'] #int(param['feature']['Fs']*0.025) # ~32 ms 24000*0.025
        self.win_length = self.n_fft
        self.Fs_Multiple = 0.5
        self.Fs = param["feature"]["Fs"]
        self.duration = param["feature"]["duration"]
        self.n_mels = param["feature"]["n_mels"]
        self.mel_length = int((self.duration*self.Fs - self.n_fft + self.hop_length)/self.hop_length)

        self.batch_size  = batch_size # divide by 2
        self.num_classes = len(self.class_dict)
        self.gen_mel = return_spec
        self.gen_stft = return_fft

        assert not (return_fft and return_spec), "Incalid configuration"

        # self.on_epoch_end()

    def toJSON(self):
        """Poor mans serialisation, I do not want to dive into Python. Please
        """
        return {
            "base_dir" : self.base_dir,
            "batch_size" : self.batch_size,
            "shuffle" : self.shuffle,
            "gentype" : self.gentype,
            "classes" : self.classes,
            "class_dict" : self.class_dict,
            "param" : self.param,
            "n_fft" : self.n_fft,
            "hop_length" : self.hop_length,
            "win_length" : self.win_length,
            "Fs_Multiple" : self.Fs_Multiple,
            "Fs" : self.Fs,
            "duration" : self.duration,
            "n_mels" : self.n_mels,
            "mel_length" : self.mel_length,
            "batch_size" : self.batch_size,
            "num_classes" : self.num_classes,
            "gen_mel" : self.gen_mel,
            "gen_stft" : self.gen_stft
        }

    def get_classes(self):
        classes = []
        for c in os.listdir(os.path.join(self.base_dir,self.gentype)):
            classes.append(c)
        return classes
    
    def get_files(self):
        lst = []
        for c in self.classes:
            tmp = glob.glob(f"{self.base_dir}/{self.gentype}/{c}/*.{self.ext}")
            # # print(tmp)
            lst += [(c, i) for i in tmp]
        return lst
    

    def __len__(self):
        return len(self.files) // self.batch_size
    
    def __getitem__(self, index):
        clip = self.clip
        files_to_return = self.files[index*self.batch_size : (index + 1)*self.batch_size]
        # # print(files_to_return)
        X = []
        Y = []
        for cls, file in files_to_return:
            audio_seg = pydub.AudioSegment.from_file(file)
            audio_seg = audio_seg.set_frame_rate(self.Fs)
            if audio_seg.__len__() < clip:
                audio_seg = audio_seg + AudioSegment.silent(duration = (clip - len(audio_seg)))
            elif audio_seg.__len__() > clip:
                audio_seg = audio_seg[:clip]
            # print(audio_seg.__len__())
            tensor = audio_seg.get_array_of_samples()
            tensor = np.array(tensor, dtype=np.float32)
            tensor = tensor/np.max(tensor)
            # print(type(tensor), tensor.shape)
            # tensor, sr = librosa.load(file, sr = self.sampling_rate)
            # # print(len(tensor))
            if self.gen_mel:
                tensor = gen_mel_feature(self.param, tensor, self.Fs, self.n_fft, self.hop_length, self.win_length, self.n_mels, log=True, window='hann', enable_librosa=False, gen_spec=True)

            elif self.gen_stft:
                tensor = gen_mel_feature(self.param, tensor, self.Fs, self.n_fft, self.hop_length, self.win_length, self.n_mels, log=True, window='hann', enable_librosa=False, gen_stft=True)
            # print(tensor.shape)
            X.append(tensor)
            Y.append(self.class_dict[cls])

            # print(len(X), len(Y))


        assert len(X) == len(Y)
        assert len(X) == self.batch_size

        X = np.array(X)
        Y = np.array(Y)

        Y = np.eye(len(self.classes))[Y]

        return X, Y


class rho_generator_audio(BaseClassificationGenerator):
    def __init__(self, param, base_dir : str, batch_size : int = 16, shuffle : bool = True, gentype : str = 'train', return_spec : bool = False, return_fft : bool = False, ext : str = 'flac', 
                 selector = None, irred_model : str = '', target_model : str = '', target_model_path : str = '', epoch_cutoff : int = 3, minibatch_size : float = 0.6, loss : tf.keras.losses.Loss = tf.keras.losses.categorical_crossentropy):
        super().__init__(param, base_dir, batch_size, shuffle, gentype, return_spec, return_fft, ext)
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