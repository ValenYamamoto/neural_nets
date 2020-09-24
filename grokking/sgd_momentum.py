import numpy as np
np.random.seed( 1 )

from keras.datasets import mnist
( xtrain, ytrain ), ( xtest, ytest ) = mnist.load_data()

images, labels = ( xtrain[ 0:1000 ].reshape( 1000, 28*28 ) / 255, ytrain[ 0:1000 ] )

one_hot_labels = np.zeros( ( len( labels ), 10 ) )
for count, label in enumerate( labels ):
    one_hot_labels[count][label] = 1
labels = one_hot_labels

test_images = xtest.reshape( len( xtest ), 28*28 ) / 255
test_labels = np.zeros( ( len( ytest ), 10 ) )
for count, label in enumerate( ytest ):
    test_labels[count][label] = 1

# using tanh instead of sigmoid because
# tanh gives values between -0.5 and 0.5
def tanh( x ):
    return np.tanh( x )

def tanh2deriv( output ):
    return 1 - ( output ** 2 )

def softmax( x ):
    temp = np.exp( x )
    return temp / np.sum( temp, axis=1, keepdims=True )

alpha = 0.01
iterations = 300
hidden_size = 100
pixels_per_image = 28*28
num_labels = 10
batch_size = 50

weights_0_1 = 0.02*np.random.random(( pixels_per_image, hidden_size )) - 0.01
weights_1_2 = 0.2*np.random.random(( hidden_size, num_labels )) - 0.1

v_last_1_2 = 0
v_last_0_1 = 0
beta = 0.9
epsilon = 1E-7

for j in range( iterations ):
    correct_cnt = 0
    for i in range( int( len( images ) / batch_size ) ):
        batch_start, batch_end = (( i * batch_size ), ( (i+1) * batch_size ))
        layer_0 = images[ batch_start:batch_end ]
        layer_1 = tanh( np.dot( layer_0, weights_0_1 ) )
        dropout_mask = np.random.randint( 2, size=layer_1.shape )
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot( layer_1, weights_1_2 ) 

        for k in range( batch_size ):
            correct_cnt += int( np.argmax( layer_2[ k:k+1 ] ) == np.argmax( labels[ batch_start+k:batch_start+k+1 ] ) )

        layer_2_delta = ( labels[ batch_start:batch_end ] - layer_2 )
        layer_1_delta = layer_2_delta.dot( weights_1_2.T ) * tanh2deriv( layer_1 )
        layer_1_delta *= dropout_mask

        v_1_2 = beta * v_last_1_2 + ( 1 - beta ) * layer_1.T.dot( layer_2_delta )
        v_1_2 /= ( 1 - beta ** j ) + epsilon
        v_last_1_2 = v_1_2
        weights_1_2 += alpha * v_1_2

        v_0_1 = beta * v_last_0_1 + ( 1 - beta ) * layer_0.T.dot( layer_1_delta )
        v_0_1 /= ( 1 - beta ** j ) + epsilon
        v_last_0_1 = v_0_1
        weights_0_1 += alpha * v_0_1

    test_correct_cnt = 0

    for i in range( len( test_images ) ):
        layer_0 = test_images[ i:i+1 ]
        layer_1 = tanh( np.dot( layer_0, weights_0_1 ) )
        layer_2 = np.dot( layer_1, weights_1_2 )
        test_correct_cnt += int( np.argmax( layer_2 ) == np.argmax( test_labels[ i:i+1 ] ) )
    
    if( j%10 == 0 ):
        print( f"I: { j }\n\tTest Acc: { test_correct_cnt/float(len(test_images)) }\n\tTrain Acc: { correct_cnt/float(len(images)) }" )
