import torch
from torchvision import datasets, transforms

class MNIST_Net( torch.nn.Module ):
    
    def __init__( self ):
        super( MNIST_Net, self ).__init__()
        self.conv = torch.nn.Conv2d( 1, 16, 3 )
        self.linear1 = torch.nn.Linear( 16*26*26, 100 )
        self.linear2 = torch.nn.Linear( 100, 10 )
        self.softmax = torch.nn.LogSoftmax( dim=1 )

    def forward( self, x ):
        conv_layer = self.conv( x )
        hidden_relu = self.linear1( conv_layer.view( 50, -1 ) ).clamp( min=0 )
        prediction = self.linear2( hidden_relu )
        prediction = self.softmax( prediction )
        return prediction

batch_size, in_dim, hidden_dim, out_dim = 50, 28*28, 100, 10

transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.1307,), (0.3081,) ) ] )

# download datasets if not already there
training_set = datasets.MNIST( "./data", train=True, download=True, transform=transform )
test_set = datasets.MNIST( "./data", train=False, download=True, transform=transform )

# get iterable data loader
train_loader = torch.utils.data.DataLoader( training_set, batch_size=batch_size )
test_loader = torch.utils.data.DataLoader( test_set, batch_size=batch_size )

device = torch.device( "cpu" )

model = MNIST_Net()
alpha = 1E-5

criterion = torch.nn.NLLLoss()
#originally Adam
optimizer = torch.optim.SGD( model.parameters(), lr=alpha )

def train( model, device, train_loader, criterion, optimizer ):
    model.train()
    for batch_id, ( data, target ) in enumerate( train_loader ):
        # data = data.view( data.shape[0], -1)
        data, target = data.to( device ), target.to( device )
        optimizer.zero_grad()
        output = model( data )
        loss = criterion( output, target )
        loss.backward()
        optimizer.step()

def test( model, device, test_loader, criterion ):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.view( data.shape[0], -1 )
            data, target = data.to( device ), target.to( device )
            output = model( data )
            test_loss += criterion( output, target )
            prediction = output.argmax( dim=1 )
            correct += prediction.eq( target.view_as( prediction ) ).sum().item()
    test_loss /= len( test_loader.dataset )

    print( f"Test set { epoch }: \n\tAverage Loss: { test_loss }\n\tAccuracy: { correct / len( test_loader.dataset ) }" )

for epoch in range( 10 ):
    train( model, device, train_loader, criterion, optimizer )
    test( model, device, test_loader, criterion )

