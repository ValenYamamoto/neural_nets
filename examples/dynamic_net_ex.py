import random
import torch

class DynamicNet( torch.nn.Module ):

    # fully connected ReLU network that on each forward pass
    # chooses a random number between 1 and 4 and uses that many
    # hidden layers, resuing the same weight multiple times
    def __init__( self, in_dim, hidden_dim, out_dim ):
        super( DynamicNet, self ).__init__()
        self.input_linear = torch.nn.Linear( in_dim, hidden_dim )
        self.middle_linear = torch.nn.Linear( hidden_dim, hidden_dim )
        self.output_linear = torch.nn.Linear( hidden_dim, out_dim )

    def forward( self, x ):
        hidden_relu = self.input_linear( x ).clamp( min=0 )
        for _ in range( random.randint( 0, 3 ) ):
            hidden_relu = self.middle_linear(hidden_relu )
        prediction = self.output_linear( hidden_relu )
        return prediction 

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim )
data_out = torch.randn( batch_size, out_dim )

model = DynamicNet( in_dim, hidden_dim, out_dim )

criterion = torch.nn.MSELoss( reduction='sum' )
optimizer = torch.optim.SGD( model.parameters(), lr=1E-4, momentum=0.9 )

for t in range( 50 ):
    prediction = model( data_in )

    loss = criterion( prediction, data_out )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
