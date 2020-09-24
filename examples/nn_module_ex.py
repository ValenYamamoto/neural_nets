import torch

batch_size, in_dim, hidden_dim, out_dim = 64, 1000, 100, 10

data_in = torch.randn( batch_size, in_dim )
data_out = torch.randn( batch_size, out_dim )

# define model as a sequence of layers
model = torch.nn.Sequential(
        torch.nn.Linear( in_dim, hidden_dim ),
        torch.nn.ReLU(),
        torch.nn.Linear( hidden_dim, out_dim ),
)

# Mean squared error loss function
loss_fn = torch.nn.MSELoss( reduction='sum' )

alpha = 1E-7
for t in range( 50 ):
    # make prediction
    prediction = model( data_in )

    loss = loss_fn( prediction, data_out )

    model.zero_grad()

    # compute gradients, all parameters are requires_grad=True
    # automatically
    loss.backward()

    with torch.no_grad():
        # get all parameters to update weights
        for param in model.parameters():
            param -= alpha * param.grad

