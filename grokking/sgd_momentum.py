import numpy as np
from keras.datasets import mnist

( x_train, y_train ), ( x_test, y_test ) = mnist.load_data()

images, labels = ( x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000] )

one_hot_labels = np.zeros(( len( labels ), 10 ))

for count, label in enumerate( labels ):
    one_hot_labels[count][label] = 1
labels = one_hot_labels

test_images = x_test.reshape( len( x_test ), 28*28 ) / 255
test_labels = np.zeros(( len( y_test ), 10 ))
for count, label in enumerate( y_test ):
    test_labels[count][label] = 1

np.random.seed(1)

relu = lambda x: (x>=0) * x
relu2deriv = lambda x : x>=0

def tanh( x ):
    x = np.clip( x, -500, 500 )
    return np.tanh( x )

tanh2deriv = lambda x: 1 - x ** 2

alpha = 0.5
iterations = 350
hidden_size = 300
pixels_per_image = 28*28
num_labels = 10

weights_0_1 = 0.2 * np.random.random(( pixels_per_image, hidden_size )) - 0.1
weights_1_2 = 0.2 * np.random.random(( hidden_size, num_labels )) - 0.1

beta = 0.95
v_last_0_1 = np.zeros(( pixels_per_image, hidden_size ))
v_last_1_2 = np.zeros(( hidden_size, num_labels ))

for j in range( iterations ):
    error, correct_cnt = 0.0, 0
    for i in range( len( images ) ):
        layer_0 = images[i:i+1]
        layer_1 = tanh( np.dot( layer_0, weights_0_1 ) )
        layer_2 = np.dot( layer_1, weights_1_2 )

        #error += np.sum( ( labels[i:i+1] - layer_2 ) ** 2 )
        correct_cnt += int( np.argmax( layer_2 ) == np.argmax( labels[i:i+1] ) )

        layer_2_delta = labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot( weights_1_2.T ) * tanh2deriv( layer_1 )

        v_0_1 = beta * v_last_0_1 + ( 1 - beta ) * layer_0.T.dot( layer_1 )
        v_0_1 /= ( 1 - beta ** (j+1) )
        v_last_0_1 = v_0_1
        v_1_2 = beta * v_last_1_2 + ( 1 - beta ) * layer_1.T.dot( layer_2 )
        v_1_2 /= ( 1 - beta ** (j+1) )
        v_last_1_2 = v_1_2
        weights_0_1 -= alpha * v_0_1
        weights_1_2 -= alpha * v_1_2

    if ( j % 10 == 0 ):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range( len( test_images ) ):
            layer_0 = test_images[i:i+1]
            layer_1 = tanh( np.dot( layer_0, weights_0_1 ) )
            layer_2 = np.dot( layer_1, weights_1_2 )

            test_error += np.sum( ( test_labels[i:i+1] - layer_2 ) ** 2 )
            test_correct_cnt += int( np.argmax( layer_2 ) == np.argmax( test_labels[i:i+1] ) )
        print( f"Error: {test_error/float(len(test_images))} Acc: {test_correct_cnt / float(len(test_images))}" )
