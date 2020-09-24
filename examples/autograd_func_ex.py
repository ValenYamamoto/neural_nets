import torch

class MyReLU( torch.autograd.Function ):

    @staticmethod
    def forward( ctx, data_in ):
        # ctx is a context object that can be used to store info for backprop
        ctx.save_for_backwards( data_in )
        return data_in.clamp( min=0 )

    @staticmethod
    def backward( ctx, grad_output ):
        # get saved input tensors from ctx
        data_in, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[ data_in<0 ] = 0
        return grad_input

dtype = torch.float
device = torch.device( "cpu" )

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim, device=device, dtype=dtype )
data_out = torch.randn( batch_size, out_dim, device=device, dtype=dtype )

w1 = torch.randn( in_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True )
w2 = torch.randn( hidden_dim, out_dim, device=device, dtype=dtype, requires_grad=True )

alpha = 1E-7
for t in range( 50 ):
    # call funciton with MyReLU.apply( input ) but here aliasing function
    relu = MyReLU.apply

    prediction = relu( data_in.mm( w1 ) ).mm( w2 )

    loss = ( prediction - data_out ).pow( 2 ).sum()

    loss.backward()

    with torch.no_grad():
        w1 -= alpha * w1.grad
        w2 -= alpha * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
