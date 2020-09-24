import torch

dtype = torch.float
device = torch.device( "cpu" )

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim, device=device, dtype=dtype )
data_out = torch.randn( batch_size, out_dim, device=device, dtype=dtype )

w1 = torch.randn( in_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True )
w2 = torch.randn( hidden_dim, out_dim, device=device, dtype=dtype, requires_grad=True )

alpha = 1E-7

for t in range( 50 ):
    # can do all in one go because don't need intermediate layers for backprop
    pred = data_in.mm( w1 ).clamp( min=0 ).mm( w2 )
    
    # calculate the error
    loss = ( pred - data_out ).pow( 2 ).sum()
    # autograd delta calculation
    loss.backward()

    # make sure that weight calculation isn't tracked
    with torch.no_grad():
        w1 -= alpha * w1.grad
        w2 -= alpha * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
