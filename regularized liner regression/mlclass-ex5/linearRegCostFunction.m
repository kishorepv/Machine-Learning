function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));


theta_tmp = zeros(size(theta));

J = (1/(2 * m)) * sum (   ((X * theta) - y ) .^ 2  ) + (lambda/(2* m)) * sum( theta(2:end) .^ 2);


theta_tmp = theta;
theta_tmp(1) = 0;

grad = ((1/m) * ((( X * theta) - y)' * X )' ) + (lambda/m) * theta_tmp;







% =========================================================================

grad = grad(:);

end
