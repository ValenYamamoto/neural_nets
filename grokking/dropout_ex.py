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

alpha = 0.005
iterations = 350
hidden_size = 40
pixels_per_image = 28*28
num_labels = 10

weights_0_1 = 0.2 * np.random.random(( pixels_per_image, hidden_size )) - 0.1
weights_1_2 = 0.2 * np.random.random(( hidden_size, num_labels )) - 0.1

for j in range( iterations ):
    error, correct_cnt = 0.0, 0
    for i in range( len( images ) ):
        layer_0 = images[i:i+1]
        layer_1 = relu( np.dot( layer_0, weights_0_1 ) )
        dropout_mask = np.random.randint( 2, size=layer_1.shape )
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot( layer_1, weights_1_2 )

        error += np.sum( ( labels[i:i+1] - layer_2 ) ** 2 )
        correct_cnt += int( np.argmax( layer_2 ) == np.argmax( labels[i:i+1] ) )

        layer_2_delta = labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot( weights_1_2.T ) * relu2deriv( layer_1 )
        layer_1_delta *= dropout_mask

        weights_0_1 += alpha * layer_0.T.dot( layer_1_delta )
        weights_1_2 += alpha * layer_1.T.dot( layer_2_delta )

    if ( j % 10 == 0 ):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range( len( test_images ) ):
            layer_0 = test_images[i:i+1]
            layer_1 = relu( np.dot( layer_0, weights_0_1 ) )
            layer_2 = np.dot( layer_1, weights_1_2 )

            test_error += np.sum( ( test_labels[i:i+1] - layer_2 ) ** 2 )
            test_correct_cnt += int( np.argmax( layer_2 ) == np.argmax( test_labels[i:i+1] ) )
        print( f"Error: {test_error/float(len(test_images))} Acc: {test_correct_cnt / float(len(test_images))}" )
