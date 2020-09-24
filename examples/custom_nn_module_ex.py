import torch

class TwoLayerNet( torch.nn.Module ):
    
    def __init__( self, in_dim, hidden_dim, out_dim ):
        super( TwoLayerNet, self ).__init__()
        self.linear1 = torch.nn.Linear( in_dim, hidden_dim )
        self.linear2 = torch.nn.Linear( hidden_dim, out_dim )

    def forward( self, x ):
        hidden_relu = self.linear1( x ).clamp( min=0 )
        prediction = self.linear2( hidden_relu )
        return prediction

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim )
data_out = torch.randn( batch_size, out_dim )

model = TwoLayerNet( in_dim, hidden_dim, out_dim )

# define loss function: mean squared error
criterion = torch.nn.MSELoss( reduction='sum' )
# optimizer SGD
optimizer = torch.optim.SGD( model.parameters(), lr=1E-7 )

for t in range( 50 ):
    prediction = model( data_in )

    loss = criterion( prediction, data_out )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
