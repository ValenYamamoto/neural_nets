import numpy as np

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

# random input output
data_in = np.random.randn( batch_size, in_dim )
data_out = np.random.randn( batch_size, out_dim )

# random weights
w1 = np.random.randn( in_dim, hidden_dim )
w2 = np.random.randn( hidden_dim, out_dim )

alpha = 1E-7

for t in range( 50 ):
    # forward pass
    hidden_layer = data_in.dot( w1 )
    hidden_relu = np.maximum( hidden_layer, 0 )
    prediction = hidden_relu.dot( w2 )

    # delta
    loss = np.square( prediction - data_out ).sum()

    # backprop
    grad_prediction = 2 * ( prediction - data_out )
    grad_w2 = hidden_relu.T.dot( grad_prediction )
    grad_hidden_relu = grad_prediction.dot( w2.T )
    grad_hidden = grad_hidden_relu.copy()
    # turn off layers if layer was turned off by relu
    grad_hidden[ hidden_layer<0 ] = 0
    grad_w1 = data_in.T.dot( grad_hidden )

    w1 -= alpha * grad_w1
    w2 -= alpha * grad_w2
