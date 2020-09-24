import torch

# type of torch tensors
dtype = torch.float
# do operations on CPU
device = torch.device( "cpu" )

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

# random input output tensors
data_in = torch.randn( batch_size, in_dim, device=device, dtype=dtype )
data_out = torch.randn( batch_size, out_dim, device=device, dtype=dtype )

# random weights
w1 = torch.randn( in_dim, hidden_dim, device=device, dtype=dtype )
w2 = torch.randn( hidden_dim, out_dim, device=device, dtype=dtype )

alpha = 1E-7

for t in range( 50 ):
    # forward prop
    hidden_layer = data_in.mm( w1 )
    hidden_relu = hidden_layer.clamp( min=0 )
    prediction = hidden_relu.mm( w2 )

    # backprop
    grad_pred = 2 * ( prediction - data_out )
    grad_w2 = hidden_relu.t().mm( grad_pred )
    grad_hidden_relu = grad_pred.mm( w2.t() )
    grad_hidden = grad_hidden_relu.clone()
    grad_hidden[ hidden_layer<0 ] = 0
    grad_w1 = data_in.t().mm( grad_hidden )

    # update weights
    w1 -= alpha * grad_w1
    w2 -= alpha * grad_w2
