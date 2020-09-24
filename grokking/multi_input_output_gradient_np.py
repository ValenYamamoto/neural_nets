import numpy as np

def neural_network( data_in, weights ):
    pred = data_in.dot( weights )
    return pred

weights = np.array( [ [ 0.1, 0.1, -0.3 ],
            [ 0.1, 0.2, 0.0 ],
            [ 0.0, 1.3, 0.1 ] ] ).T

toes = [ 8.5, 9.5, 9.9, 9.0 ]
wlrec = [ 0.65, 0.8, 0.8, 0.9 ]
nfans = [ 1.2, 1.3, 0.5, 1.0 ]

hurt = [ 0.1, 0.0, 0.0, 0.1 ]
win = [ 1, 1, 0, 1 ]
sad = [ 0.1, 0.0, 0.1, 0.2 ]

data_in = np.array( [ toes[0], wlrec[0], nfans[0] ] )
true = [ hurt[0], win[0], sad[0] ]

pred = neural_network( data_in, weights )

error = [ ( pred[i] - true[i] ) ** 2 for i in range( len( pred ) ) ]
delta = [ pred[i] - true[i] for i in range( len( pred ) ) ]

weight_delta = np.outer( data_in, delta ).T

print(weights )

alpha = 0.01
weights = [ [ weights[i][j] - ( weight_delta[i][j] * alpha ) for j in range( len( weights[0] ) ) ] for i in range( len( weights ) ) ]

print( weights )
print( weight_delta )
