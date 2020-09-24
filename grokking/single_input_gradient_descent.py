weight = 0.5
goal_pred = 0.8
data_in = 2
alpha = 0.1

for i in range( 20 ):
    pred = data_in * weight
    error = ( pred - goal_pred ) ** 2
    derivative = data_in * ( pred - goal_pred )
    weight -= alpha * derivative

    print( f"Error: {error} Prediction: {pred}" )
