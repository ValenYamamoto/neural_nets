def vect_mat_mul( vect: list, matrix: [list] ):
	output = [ sum( vect[i] * matrix[j][i] for i in range( len( vect ) ) ) for j in range( len( matrix ) ) ]
	return output

def neural_network( data: list, weights: list ):
	pred = vect_mat_mul( data, weights )
	return pred

weights = [ [ 0.1, 0.1, -0.3 ],
            [ 0.1, 0.2, 0.0 ],
            [ 0.0, 1.3, 0.1 ] ]

toes = [ 8.5, 9.5, 9.9, 9.0 ]
wlrec = [ 0.65, 0.8, 0.8, 0.9 ]
nfans = [ 1.2, 1.3, 0.5, 1.0 ]

data_in = [ toes[0], wlrec[0], nfans[0] ]

pred = neural_network( data_in, weights )

print( pred )
