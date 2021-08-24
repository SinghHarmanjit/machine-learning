function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hValue = sigmoid(X*theta);
sum = 0;
j = 1;

% cost function calculations
count = size(X)(1);
for i = 1:count
    sum = sum + -1*y(i,j)*log(hValue(i,j)) - (1 -1*y(i,j))*log(1 - hValue(i,j));
endfor
J = sum/count;

% grad calculations
for z = 1: size(grad)(1)
    sum = 0;
    for i = 1:count
        sum = sum + (hValue(i,j) - y(i,j))*X(i, z);
    endfor
    grad(z,1) = sum/m;
endfor





% =============================================================

end
