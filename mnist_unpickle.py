import torch
import pickle

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

def create_one_hot_labels( data, batch_size ):
    one_hot_labels = torch.zeros( [ batch_size, 10 ], dtype=torch.float )
    for count in range( len( data ) ):
        label = data[ count ]
        one_hot_labels[count][label] = 1
    return one_hot_labels


batch_size = 50

model = MNIST_Net()
model.load_state_dict( torch.load( "./pickles/mnist_net_state_dict.p" ) )

transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.1307,), (0.3081,) ) ] )

# download datasets if not already there
test_set = datasets.MNIST( "./data", train=False, download=True, transform=transform )

# get iterable data loader
test_loader = torch.utils.data.DataLoader( test_set, batch_size=batch_size )

device = torch.device( "cpu" )

criterion = torch.nn.MSELoss()

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

    print( f"Test set\n\tAverage Loss: { test_loss }\n\tAccuracy: { correct / len( test_loader.dataset ) }" )

test( model, device, test_loader, criterion )

