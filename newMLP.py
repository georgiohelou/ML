import pandas as pd
from sklearn.neural_network import MLPRegressor

# Create Neural Net MLP regressor 
# Explore settings logarithmically (0.1, 0.01, 0.001, 0.00001)
model = MLPRegressor(
    # try some layer & node sizes
    hidden_layer_sizes=(5,17), 
    # find a learning rate?
    learning_rate_init=.001,
    # activation functions (relu, tanh, identity)
    activation='relu',
    max_iter=2000
);

# Train it (where X_train is your feature matrix & Y_train is a vector with desired target values for each record)
nn_regr.fit(X_train,Y_train)

# Plot the 'loss_curve_' protery on model to see how well we are learning over the iterations
# Use Pandas built in plot method on DataFrame to creat plot in one line of code
pd.DataFrame(model.loss_curve_).plot()