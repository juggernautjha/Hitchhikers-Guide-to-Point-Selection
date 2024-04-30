from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation, Flatten, Conv1D, Add, Multiply, Lambda, Conv2DTranspose, Concatenate, UpSampling2D, Reshape, Dot, Permute, RepeatVector


def BaseCNN(Width, Height, num_classes, Channel=1) :
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(Width,Height,Channel)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model




# def CNN_enc(width, height, num_classes, filters = [(128, (5,5)), (64, (3,3)),  (64, (3,3)),  (32, (3,3))], channel=1):
#     input = Input(shape=(width, height, channel))
    
#     x = input
#     for i in range(len(filters)):
#         x = Conv2D(filters[i][0], filters[i][1], activation='relu')(x)
#         x = MaxPooling2D(pool_size=(2,2))(x)
#         x = BatchNormalization()(x)
    
#     x = Dropout(0.25)(x)
    
#     x = Flatten()(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.25)(x)
#     x = Dense(num_classes, activation='softmax')(x)
    
#     return Model(input, x)





def CNN_enc(Width, Height, num_classes,  filters = [(128, (5,5)), (64, (3,3)),  (64, (3,3)),  (32, (3,3))], channel=1) :
    model = Sequential()
    
    model.add(Conv2D(filters[0][0], filters[0][1], activation='relu', input_shape=(Width,Height,channel)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    for i in range(1, len(filters)):
        model.add(Conv2D(filters[i][0], filters[i][1], activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(MaxPooling2D(pool_size=(2,2))
        model.add(BatchNormalization())
    # model.add(Conv2D(16, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(8, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model


from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Attention, Layer, Activation, Flatten, Conv1D, Add, Multiply, Lambda, Conv2DTranspose, Concatenate, UpSampling2D, Reshape, Dot, Permute, RepeatVector
import tensorflow as tf
from keras_self_attention import SeqSelfAttention

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
                             activation='relu', name='skip_'+name)(mult)

        return transformed, skip_out
      
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


def WavenetEncoder(width, height, hidden_channels = [48, 24, 12, 6], num_layer=4, kernel_size=3) :
    input_shape = (width,height)
    io_channels = height
    #hidden_channels = [48, 24, 12, 6]
    #num_layer = 4

    x = inputLayer = Input(shape=input_shape, name='MelInput')
    x = BatchNormalization()(x)

    # 'Resize' to make everything 'io_channels' big at the layer interfaces
    x = s0 = Conv1D(io_channels, 1, 
                    padding='same', use_bias=True, 
                    activation='linear', name='mel_log_expanded')(x)

    s = list()
    for i in range(num_layer) :
        dilation = 2 ** i
        x,s_tmp = wavenet_layer(channels=hidden_channels[i], hidden_channels=hidden_channels[i], kernel_size=kernel_size,  dilation_rate=dilation, name=f"{i}_e")(x)
        #x,s_tmp = wavenet_layer(io_channels, hidden_channels=hidden_channels[i], kernel_size=kernel_size,  dilation_rate=dilation, name=f"{i}_e")(x)
        x = BatchNormalization()(x)
        #x = SeqSelfAttention(attention_activation='sigmoid')(x)
        s_tmp = BatchNormalization()(s_tmp)
        s.append(s_tmp)
    s.append(x)

    x = Concatenate( axis=-1 )(s) #[x, s[num_layer-1], s[0], s[int(num_layer/2)]] )
    #x = Add(name="e_add")(s)
    x = Activation('relu', name='e_add_out')(x)
    #x = SeqSelfAttention(attention_activation='sigmoid')(x)

    # Bottleneck ts, 128 -> ts x 8
    x = last_wavenet_layer(hidden_channels[-1], hidden_channels[-1]*1, 1, dilation, 'b2')(x)
    #x = Activation('relu', name="emb_bl")(x)
    #x = Conv1D(1, kernel_size, padding='causal',  activation='relu', data_format='channels_first')(x)
    #x = Activation('linear', name="emb_bl")(x)
    x = Flatten(name="flat")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)

    return Model(inputLayer, x, name='WavenetEncoder')

def WavenetClassifier(width, height, num_classes, hidden_channels = [48, 24, 12, 6], num_layer=4, kernel_size=8, num_peaks=1) :
    if num_peaks <= 1:
        input_shape = (width,height)
    else :
        input_shape = (num_peaks, width, height)

    x = inputLayer = Input(shape=input_shape, name='MelInput')
    x = BatchNormalization()(x)

    cm = WavenetEncoder(width, height, hidden_channels= hidden_channels, num_layer= num_layer, kernel_size= kernel_size)
    #cm.summary()

    projs = []
    for i in range(num_peaks):
        # Slicing the ith channel:
        if num_peaks > 1:
            input_slice = Lambda(lambda x: x[:, i, :])(inputLayer)
        else :
            input_slice = inputLayer

        # Setting up your per-channel layers (replace with actual sub-models):
        x = cm(input_slice)
        projs.append(x)

    if num_peaks > 1:
        x = Concatenate()(projs)        
        x = Dense(128, activation='relu')(x)
    
    x = Dropout(0.25)(x)    
    x = Dense(num_classes, activation='softmax')(x)
        
    model = Model(inputs=inputLayer, outputs=x)
    return model