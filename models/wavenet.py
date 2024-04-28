from tensorflow.keras import Model
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, Conv1D, Add, Multiply, Lambda, Conv2DTranspose, Concatenate, UpSampling2D, Reshape, Dot, Permute, RepeatVector
import tensorflow as tf



#! Split the autoencoder into encoder and decoder, chuck the decoder
def AutoEncoderWavenetBase(n_mels, frames, kernel_size = 16, hidden_channels = [128, 32, 8, 2, 1], beta=0):

    def wavenet_layer(channels, hidden_channels, kernel_size, dilation_rate, name):
        def f(input_):
            filter_out = Conv1D(hidden_channels, kernel_size,
                            strides=1, dilation_rate=dilation_rate,
                            padding='causal', use_bias=True, 
                            activation='tanh', name='filter_'+name)(input_)
            gate_out   = Conv1D(hidden_channels, kernel_size,
                            strides=1, dilation_rate=dilation_rate,
                            padding='causal', use_bias=True, 
                            activation='sigmoid', name='gate_'+name)(input_)
            mult = Multiply(name='mult_'+name)( [filter_out, gate_out] )
            
            transformed = Conv1D(channels, 1, 
                                padding='same', use_bias=True, 
                                activation='linear', name='trans_'+name)(mult)
            skip_out    = Conv1D(channels, 1, 
                                padding='same', use_bias=True, 
                                activation='leaky_relu', name='skip_'+name)(mult)

            return Add(name='resid_'+name)( [transformed, input_] ), skip_out
        
        return f

    def last_wavenet_layer(channels, hidden_channels, kernel_size, dilation_rate, name):
        def f(input_):

            filter_out = Conv1D(hidden_channels, kernel_size,
                            strides=1, dilation_rate=dilation_rate,
                            padding='causal', use_bias=True, 
                            activation='tanh', name='filter_'+name)(input_)
            gate_out   = Conv1D(hidden_channels, kernel_size,
                            strides=1, dilation_rate=dilation_rate,
                            padding='causal', use_bias=True, 
                            activation='sigmoid', name='gate_'+name)(input_)
            mult = Multiply(name='mult_'+name)( [filter_out, gate_out] )

            return mult
        
        return f

    def Conv1DTranspose(input_tensor, filters, kernel_size, strides=1, padding='same', dilation_rate=1, use_bias=True, activation='linear', name='layer'):
        x = Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), 
                            padding=padding, name=name)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x


    def decode_wavenet_layer(channels, hidden_channels, kernel_size, dilation_rate, name):
        def f(input_):
            filter_out = Conv1DTranspose(input_, hidden_channels, kernel_size,
                                        strides=1, padding='same', dilation_rate=dilation_rate, use_bias=True, 
                                        activation='tanh', name='filter_'+name)
            gate_out   = Conv1DTranspose(input_, hidden_channels, kernel_size,
                                        strides=1, 
                                        padding='same', dilation_rate=dilation_rate, use_bias=True, 
                                        activation='sigmoid', name='gate_'+name)
            mult       = Multiply(name='mult_'+name)( [filter_out, gate_out] )
            
            transformed = Conv1DTranspose(mult, channels, 1, 
                                            padding='same', use_bias=True, 
                                            activation='linear', name='trans_'+name)
            skip_out    = Conv1DTranspose(mult, channels, 1, 
                                            padding='same', use_bias=True, 
                                            activation='leaky_relu', name='skip_'+name)
            
            return Add(name='resid_'+name)( [transformed, input_] ), skip_out
        
        return f

    def get_model1(n_mels, ts, kernel_size = 16, hidden_channels = [128, 32, 8, 2, 1]) :
        """
        Default mSense model with MSE
        """
        
        input_shape = (ts,n_mels)
        io_channels = n_mels
        num_layer = len(hidden_channels)

        x = inputLayer = Input(shape=input_shape, name='MelInput')
        x = BatchNormalization()(x)

        # 'Resize' to make everything 'io_channels' big at the layer interfaces
        x = s0 = Conv1D(io_channels, 1, 
                        padding='causal', use_bias=True, 
                        activation='linear', name='mel_log_expanded')(x)
        
        s = list()
        for i in range(num_layer) :
            dilation = 2 ** i
            x,s_tmp = wavenet_layer(io_channels, hidden_channels[i], kernel_size,  dilation, 'e'+str(i))(x)
            x = BatchNormalization()(x)
            s_tmp = BatchNormalization()(s_tmp)
            s.append(s_tmp)

        x = Add(name="e_add")(s)
        x = Activation('relu', name='e_add_out')(x)


        x = last_wavenet_layer(io_channels, hidden_channels[-1]*1, 1, dilation, 'b2')(x)
        x = Flatten()(x)
        x = BatchNormalization(name="norm_bl")(x)
        x = Activation('relu', name="emb_bl")(x)

        return Model(inputs=inputLayer, outputs=x)

    return get_model1(n_mels, frames, kernel_size, hidden_channels)