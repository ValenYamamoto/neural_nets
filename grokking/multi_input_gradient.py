def neural_network( data_in, weights ) :
    pred = sum( data_in[ i ] * weights[ i ] for i in range( len( data_in ) ) )
    return pred

weights = [ 0.1, 0.2, -0.1 ]
toes = [ 8.5, 9.5, 9.9, 9.0 ]
wlrec = [ 0.65, 0.8, 0.8, 0.9 ]
nfans = [ 1.2, 1.3, 0.5, 1.0 ]

win_or_lose_binary = [ 1, 1, 0, 1 ]

true = win_or_lose_binary[ 0 ]
data_in = [ toes[0], wlrec[0], nfans[0] ]

pred = neural_network( data_in, weights )

error = ( pred - true ) ** 2
delta = pred - true

weight_deltas = [ delta * data_in[i] for i in range( len( data_in ) ) ]

alpha = 0.01

weights = [ weights[i] - ( alpha * weight_deltas[i]) for i in range( len( weight_deltas ) ) ]

print( weights)
print( weight_deltas )
