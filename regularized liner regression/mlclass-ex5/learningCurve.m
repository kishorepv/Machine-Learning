function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

m = size(X, 1);
n = size(X, 2)

train_theta = zeros(size(X,2),1);
for i = 1 : m,

train_theta = 	trainLinearReg (X (1:i, :), y (1:i), lambda);

[error_train (i)] = linearRegCostFunction(X (1:i, :), y (1:i) , train_theta, 0 );
[error_val (i)] =  linearRegCostFunction(Xval, yval, train_theta, 0 );
 

end


end
