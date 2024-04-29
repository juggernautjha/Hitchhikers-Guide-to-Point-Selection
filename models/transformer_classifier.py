from tensorflow.keras import Model, initializers
from tensorflow.keras import layers
import tensorflow as tf

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def wavenet_layer(channels, hidden_channels, stride, kernel_size, dilation_rate, name):
    def f(input_):
        filter_out = layers.Conv1D(hidden_channels, kernel_size,
                          strides=stride, dilation_rate=dilation_rate,
                          padding='causal', use_bias=True, 
                          activation='tanh', name='filter_'+name)(input_)
        gate_out   = layers.Conv1D(hidden_channels, kernel_size,
                          strides=stride, dilation_rate=dilation_rate,
                          padding='causal', use_bias=True, 
                          activation='sigmoid', name='gate_'+name)(input_)
        mult = layers.Multiply(name='mult_'+name)( [filter_out, gate_out] )
        
        transformed = layers.Conv1D(channels, 1, 
                             strides=1, padding='same', use_bias=True, 
                             activation='linear', name='trans_'+name)(mult)
        skip_out    = layers.Conv1D(channels, 1, 
                             strides=1, padding='same', use_bias=True, 
                             activation='relu', name='skip_'+name)(mult)

        return transformed, skip_out
      
    return f


def BaseTransformerClassifier(width, height, num_classes) : #, Fs=16000) :
    input_shape = (width,height)

    size_10ms = height #Fs//100

    input = tf.keras.Input(shape=input_shape)

    # Few iterations to try
    # Change filters and kernal sizes

    # 10 ms
    x = layers.Conv1D(16, kernel_size=size_10ms, strides=1, dilation_rate=1, activation='relu', padding='same', kernel_initializer='he_uniform') (input)
    x = layers.Reshape((-1, 16))(x)
    #x = layers.Conv1D(16, kernel_size=size_10ms, strides=4, dilation_rate=1, activation='relu', padding='same', kernel_initializer='he_uniform') (input)
    #x = layers.Permute((2,1))(x)
    #x, _ = wavenet_layer(16, 16, 4, 8, 1, "e0") (x)
    #x, _ = wavenet_layer(16, 16, 4, 8, 1, "e1") (x)

    s = list()
    hidden_channels = [32, 4]
    for i in range(2) :
        dilation = 1# 2 ** (i+1)
        stride = 2**(i+1)
        x,s_tmp = wavenet_layer(hidden_channels[i], hidden_channels[i], stride,  8,  dilation, 'e'+str(i))(x)
        x = layers.BatchNormalization()(x)
        s_tmp = layers.BatchNormalization()(s_tmp)
        s.append(s_tmp)
        x = layers.Dropout(0.2)(x)


    #x = layers.Concatenate( axis=-1 )( [x, s[0]] )

    #for i in [1, 2, 4]:
    i=1
    x = transformer_encoder(x, 128//i, 8//i, 8//i, 0.2)

    x = layers.Reshape((x.shape[1]*x.shape[2], 1))(x)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in [128]:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(input, outputs)
