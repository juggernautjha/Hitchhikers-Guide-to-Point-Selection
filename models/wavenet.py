from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, Conv1D, Add, Multiply, Lambda, Conv2DTranspose, Concatenate, UpSampling2D, Reshape, Dot, Permute, RepeatVector
import tensorflow as tf

def AutoEncoderWavenetBase(n_mels, frames, kernel_size = 16, hidden_channels = [128, 32, 8, 2, 1], beta=0):
    ########################################################################
    # Wavenet layer - used by mSense model
    ########################################################################

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

    # Upsample Layer
    def upsample_layer (hidden_channels, kernel_size, scale, suffix) :
        def f(x) :
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
            x = Conv2DTranspose(filters=hidden_channels*scale, kernel_size=(kernel_size, 1), strides=(1, 1),
                                padding='same', name="u" + str(suffix))(x)
            x = Lambda(lambda x: K.squeeze(x, axis=1))(x)

            x,s2 = decode_wavenet_layer(hidden_channels*scale, hidden_channels*scale, kernel_size, 1, 'd'+str(suffix))(x)

            return x

        return f

    def upsample_layer_vae (hidden_channels, kernel_size, scale, suffix) :
        def f(x) :
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
            x = Lambda(lambda x: K.expand_dims(x, axis=1))(x)
            x = Conv2DTranspose(filters=hidden_channels*scale, kernel_size=(kernel_size, 1), strides=(1, 1),
                                padding='same', name="u" + str(suffix))(x)
            x = Lambda(lambda x: K.squeeze(x, axis=1))(x)

            x,s2 = decode_wavenet_layer(hidden_channels*scale, hidden_channels*scale, kernel_size, 1, 'd1'+str(suffix))(x)

            return x

        return f

    ########################################################################


    ########################################################################
    # Custom loss function
    ########################################################################
    beta = K.variable(value=beta)

    def customLoss(y_true,y_pred):
        diff = y_true - y_pred
        mse = K.sum(K.square(diff), axis=(1,2))/(n_mels*frames) # contains MSE for each sample

        #tf.print (K.shape(diff), K.shape(mse))
        mse_var = K.square(K.std(mse))
        sigma = K.std(mse)
        #mse_var = K.sum(K.square(K.std(diff, axis=0, keepdims=True))) # contains MSE for each feature
        #mse_var = K.sum(K.square(diff), axis=0)/n_mels # contains MSE for each feature
        #tf.print (mse_var, K.shape(mse_var))

        mse_mean = K.mean(mse)
        good_mse_mean = 0.0
        #good_mse_var = 0.0
        #mse_var = K.mean(mse_var)
        #tf.print(mse_mean, mse_var)

        # tf.print(" ", mse_mean, good_mse_mean, good_mse_var, mse_var)

        if beta != 0.0:
            gt_sigma = mse > (mse_mean + 2.5*sigma)  # >1% of data is anomalous
            gt_sigma = K.cast(gt_sigma, dtype='float32')

            good_label = 1 - gt_sigma   # gt_sigma is true when sample is abnormal

            # apply narrow std for good labels
            good_mse      = good_label*mse
            good_mse_mean = K.sum(good_mse)/K.cast(tf.math.count_nonzero(good_mse), tf.float32)
            #good_mse_var  = K.square(K.std(good_mse))

            # Ignore Bad sample
            mse_mean = good_mse_mean #+ beta*good_mse_var

        #tf.print(" ", mse_mean, good_mse_mean, good_mse_var, mse_var)

        return  mse_mean + beta*mse_var

    ########################################################################
    # Warmup function used by customLoss function
    ########################################################################
    def warmup(epoch):
        value1 = 0.0 if epoch < 50 else 3
        #print("beta:", value1)
        K.set_value(beta, value1)

    wu_cb = LambdaCallback(on_epoch_end=lambda epoch, log: warmup(epoch))

    ########################################################################
    # keras model 1
    ########################################################################
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

        #x = Conv1D(1, 1, padding='same', use_bias=True, activation='relu', name='e_out')(x)
        #x = Conv1D(1, 1, padding='same', use_bias=True, activation='linear', name='emb_bl')(x)


        # Bottleneck ts, 128 -> ts x 8
        x = last_wavenet_layer(io_channels, hidden_channels[-1]*1, 1, dilation, 'b2')(x)
        x = Flatten()(x)
        x = BatchNormalization(name="norm_bl")(x)
        x = Activation('relu', name="emb_bl")(x)

        scale = int(io_channels/hidden_channels[-1])
        x = Reshape((ts, -1))(x)
        x = upsample_layer(hidden_channels[-1], kernel_size=1, scale=scale, suffix=1)(x)

        inverselist = hidden_channels[::-1]
        io_channels = n_mels
        for i in range(num_layer) :
            dilation = 2 ** i
            x,s_tmp = wavenet_layer(io_channels, inverselist[i], kernel_size,  dilation, 'w'+str(i))(x)
            x = BatchNormalization()(x)
            s_tmp = BatchNormalization()(s_tmp)
            s.append(s_tmp)

        return Model(inputs=inputLayer, outputs=x)

    return get_model1(n_mels, frames, kernel_size, hidden_channels), customLoss, wu_cb