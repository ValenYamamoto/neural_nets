import torch
import pickle

import torch.nn.functional as F

from torchvision import datasets, transforms

def create_one_hot_labels( data, batch_size ):
    one_hot_labels = torch.zeros( [ batch_size, 10 ], dtype=torch.float )
    for count in range( len( data ) ):
        label = data[ count ]
        one_hot_labels[count][label] = 1
    return one_hot_labels

class MNIST_Net( torch.nn.Module ):
    
    def __init__( self ):
        super( MNIST_Net, self ).__init__()
        self.conv = torch.nn.Conv2d( 1, 16, 3 )
        self.pooling = torch.nn.MaxPool2d( 3, stride=1 )
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear( 16*24*24, 100 )
        self.linear2 = torch.nn.Linear( 100, 10 )
        self.softmax = torch.nn.LogSoftmax( dim=1 )

    def forward( self, x ):
        conv_layer = F.relu( self.conv( x ) )
        pool = self.pooling( conv_layer )
        flat = self.flatten( pool )
        hidden_relu = F.relu( self.linear1( flat ) )
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

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=alpha )

def train( model, device, train_loader, criterion, optimizer ):
    model.train()
    for batch_id, ( data, target ) in enumerate( train_loader ):
        # data = data.view( data.shape[0], -1)
        target = create_one_hot_labels( target, batch_size )
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
            labels = target
            target = create_one_hot_labels( target, batch_size )
            data, target = data.to( device ), target.to( device )
            output = model( data )
            test_loss += criterion( output, target ).sum().item()
            prediction = output.argmax( dim=1 )
            correct += prediction.eq( labels.view_as( prediction ) ).sum().item()
    test_loss /= len( test_loader.dataset )

    print( f"Test set { epoch }: \n\tAverage Loss: { test_loss }\n\tAccuracy: { correct / len( test_loader.dataset ) }" )

for epoch in range( 10 ):
    train( model, device, train_loader, criterion, optimizer )
    test( model, device, test_loader, criterion )

pickle.dump( MNIST_Net, open( "./pickles/mnist_net_pickle.p", "wb" ) )
torch.save( model.state_dict(), "./pickles/mnist_net_state_dict.p" )
