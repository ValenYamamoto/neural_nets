def neural_network( data_in, weight ):
    prediction = data_in * weight
    return prediction

weight, goal_pred, data_in = ( 0.0, 0.8, 1.1 )

for i in range( 4 ):
    print( f"----\nWeight: {weight}" )

    pred = neural_network( data_in, weight )

    error = ( pred - goal_pred ) ** 2

    delta = pred - goal_pred

    weight_delta = data_in * delta

    weight -= weight_delta 

    print( f"Error: {error} Prediction: {pred}" )
    print( f"Delta: {delta} Weight Delta: {weight_delta}" )
