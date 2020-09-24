def neural_network( data_in, weights ) :
    pred = [ data_in * weights[i] for i in range( len( weights ) ) ]
    return pred

weights = [ 0.3, 0.2, 0.9 ]

wlrec = [ 0.65, 1.0, 1.0, 0.9 ]

hurt = [ 0.1, 0.0, 0.0, 0.1 ]
win = [ 1, 1, 0, 1 ]
sad = [ 0.1, 0.0, 0.1, 0.2 ]

data_in = wlrec[0]
true = [ hurt[0], win[0], sad[0] ]

pred = neural_network( data_in, weights )

error = [ ( pred[i] - true[i] ) ** 2 for i in range( len( pred ) ) ]
delta = [ pred[i] - true[i] for i in range( len( pred ) ) ]

weight_deltas = [ data_in * delta[i] for i in range( len( delta ) ) ]

alpha = 0.1

weights = [ weights[i] - ( alpha * weight_deltas[i] ) for i in range( len( weights ) ) ]

print( weights )
print( weight_deltas )
