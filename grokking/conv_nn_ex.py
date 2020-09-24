import numpy as np
from keras.datasets import mnist

np.random.seed( 1 )

( xtrain, ytrain ), ( xtest, ytest ) = mnist.load_data()

images, labels = ( xtrain[ 0:1000 ].reshape( 1000, 28*28 ) / 255, ytrain[ 0:1000 ] )

one_hot_labels = np.zeros(( len( labels ), 10 ))
for count, label in enumerate( labels ):
    one_hot_labels[count][label] = 1
labels = one_hot_labels

test_images = xtest.reshape( len( xtest ), 28*28 ) / 255
test_labels = np.zeros(( len( ytest ), 10 ))
for count, label in enumerate( ytest ):
    test_labels[count][label] = 1

def tanh( x ):
    return np.tanh( x )

def tanh2deriv( x ):
    return 1 - ( x ** 2 )

def softmax( x ):
    temp = np.exp( x )
    return temp / np.sum( temp, axis=1, keepdims=True )

alpha = 2
iterations = 300
pixels_per_images = 784
num_labels = 10
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

# 25 * 25 *16
hidden_size = (( input_rows - kernel_rows ) * ( input_cols - kernel_cols )) * num_kernels

# ( 3*3, 16 )
kernels = 0.02*np.random.random(( kernel_rows * kernel_cols, num_kernels )) - 0.01

# ( 25*25*16, 10 )
weights_1_2 = 0.2*np.random.random(( hidden_size, num_labels )) - 0.1

def get_image_section( layer, row_from, row_to, col_from, col_to ):
    section = layer[ :, row_from:row_to, col_from:col_to ]
    return section.reshape( -1, 1, row_to - row_from, col_to - col_from )

for j in range( iterations ):
    correct_cnt = 0
    for i in range( int(len( images ) / batch_size ) ):
        batch_start, batch_end = (( i * batch_size ), ( (i+1) * batch_size ) )
        layer_0 = images[ batch_start:batch_end ]
        # (128, 28, 28)
        layer_0 = layer_0.reshape( layer_0.shape[0], 28, 28 )
        
        sects = list()
        for row_start in range( layer_0.shape[1] - kernel_rows ):
            for col_start in range( layer_0.shape[2] - kernel_cols ):
                sect = get_image_section( layer_0, row_start, row_start+kernel_rows, col_start, col_start+kernel_cols )
                sects.append( sect )

        # each image 625 3*3 sects x 128 images per batch
        expanded_input = np.concatenate( sects, axis=1 )
        es = expanded_input.shape # (128, 625, 3, 3)
        flattened_input = expanded_input.reshape( es[0]*es[1], -1 ) # (80000, 9)

        kernel_output = flattened_input.dot( kernels ) # 80000x16
        layer_1 = tanh( kernel_output.reshape( es[0], -1 ) )
        dropout_mask = np.random.randint( 2, size=layer_1.shape )
        layer_1 *= dropout_mask * 2
        layer_2 = softmax( np.dot( layer_1, weights_1_2 ))

        for k in range( batch_size ):
            labelset = labels[ batch_start+k:batch_start+k+1 ]
            _inc = int( np.argmax( layer_2[ k:k+1 ] ) == np.argmax( labelset ) )
            correct_cnt += _inc

        layer_2_delta = ( labels[ batch_start:batch_end ] - layer_2  ) / ( batch_size * layer_2.shape[0] )
        layer_1_delta = layer_2_delta.dot( weights_1_2.T ) * tanh2deriv( layer_1 )
        layer_1_delta *= dropout_mask
        weights_1_2 += alpha * layer_1.T.dot( layer_2_delta )
        l1d_reshape = layer_1_delta.reshape( kernel_output.shape )
        k_update = flattened_input.T.dot( l1d_reshape )
        kernels -= alpha * k_update

    test_correct_cnt = 0

    for i in range( len( test_images ) ):
        layer_0 = test_images[ i:i+1 ]
        layer_0 = layer_0.reshape( layer_0.shape[0], 28, 28 )
        layer_0.shape

        sects = list()
        for row_start in range( layer_0.shape[1] - kernel_rows ):
            for col_start in range( layer_0.shape[2] - kernel_cols ):
                sect = get_image_section( layer_0, row_start, row_start+kernel_rows, col_start, col_start+kernel_cols )
                sects.append( sect )
        expanded_input = np.concatenate( sects, axis=1 )
        es = expanded_input.shape
        flattened_input = expanded_input.reshape( es[0]*es[1], -1 )

        kernel_output = flattened_input.dot( kernels )
        layer_1 = tanh( kernel_output.reshape( es[0], -1 ) )
        layer_2 = np.dot( layer_1, weights_1_2 )

        test_correct_cnt += int( np.argmax( layer_2 ) == np.argmax( test_labels[ i:i+1 ] ) )

    if ( j%10 == 0 ):
        print( f"I: {j}\n\tTest Acc: { test_correct_cnt/float(len(test_images)) }\n\tTrain Acc: { correct_cnt/float(len(images)) }" )
