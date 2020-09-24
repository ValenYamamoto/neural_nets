import numpy as np
from keras.datasets import mnist

( x_train, y_train ),  ( x_test, y_test ) = mnist.load_data()

images, labels = ( x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000] )

# create answer vectors for training
one_hot_labels = np.zeros ( ( len(labels), 10 ) )
for count, label in enumerate( labels ):
    one_hot_labels[count][label] = 1
labels = one_hot_labels

# create test inputs
test_images = x_test.reshape( len( x_test ), 28*28 ) / 255

# create answer vectors for testing
test_labels = np.zeros( ( len( y_test ), 10 ) )
for count, label in enumerate( y_test ):
    test_labels[count][label] = 1

# random weights 28x28 (size of picture) 10 (size of output vector)
np.random.seed(1)
weights = 2 * np.random.rand( 28*28, 10 ) - 1

# neural network function
def neural_network( data_in, weights ):
    pred = data_in.dot( weights )
    return pred

def find_max( output ):
    return sorted( enumerate( output ), key=lambda x : -x[1] )[0][0]

for _ in range( 300 ):
    for image, label in zip( images, labels ):
        data_in = image
        true = label

        pred = neural_network( data_in, weights )

        error = [ ( pred[i] - true[i] ) ** 2 for i in range( len( pred ) ) ]
        delta = [ pred[i] - true[i] for i in range( len( pred ) ) ]

        weight_delta = np.outer( data_in, delta )

        alpha = 0.01

        weights = [ [ weights[i][j] - ( weight_delta[i][j] * alpha ) for j in range( len( weights[0] ) ) ] for i in range( len( weights ) ) ]

print( "Finished Testing" )

correct = 0
for test, label in zip( test_images, test_labels ):
    data_in = test
    true = label

    pred = neural_network( data_in, weights )

    prediction = find_max( pred )
    actual = find_max( true )
    if prediction == actual:
        correct += 1

    # print( f"Pred: { prediction } Actual: { actual }" )
print( f"Percent Correct: { correct / len( test_images ) }" )

