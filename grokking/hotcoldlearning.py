weight = 0.5
data_in = 0.5
goal_prediction = 0.8

step_amount = 0.001

for i in range( 1101 ):
	prediction = data_in * weight
	error = ( prediction - goal_prediction ) ** 2
	
	print( f"Error: {error} Prediction: {prediction}" )

	up_pred = data_in * ( weight + step_amount )
	up_error = ( goal_prediction - up_pred ) ** 2
	
	down_pred = data_in * ( weight - step_amount )
	down_error = ( goal_prediction - down_pred ) ** 2

	if ( down_error < up_error ):
		weight = weight - step_amount
	if ( down_error > up_error ):
		weight = weight + step_amount
