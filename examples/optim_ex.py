import torch

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim )
data_out = torch.randn( batch_size, out_dim )

model = torch.nn.Sequential(
        torch.nn.Linear( in_dim, hidden_dim ),
        torch.nn.ReLU(),
        torch.nn.Linear( hidden_dim, out_dim ),
)

loss_fn = torch.nn.MSELoss( reduction='sum' )

alpha = 1E-7
# use optim package to define an Optimizer that willl update weights
# for us. Adam is an optimization algorithm
optimizer = torch.optim.Adam( model.parameters(), lr=alpha )

for t in range( 50 ):
    prediction = model( data_in )

    loss = loss_fn( prediction, data_out )

    optimizer.zero_grad()

    loss.backward()

    # calling step function updates parameters
    optimizer.step()
