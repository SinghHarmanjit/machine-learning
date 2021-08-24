function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

yVector = zeros(m, num_labels);
for(index = 1:m)
    yVector(index, y(index)) = 1;
endfor

cost = 0;
for (indexI = 1:m)
    XofI1 = [ones(1,1) X(indexI,:)];
    h1 = sigmoid(Theta1*XofI1');

    XofI2 = [ones(1,1); h1];

    h2 = sigmoid(Theta2*XofI2);
    yOfI = yVector(indexI,:)';

    costOfI =  -1*log(h2')*yOfI + (log(1 - h2))'*(yOfI - 1);
    cost = cost + costOfI;
endfor

regularizationTotal = 0;
for (indexI = 1:size(Theta1,1))
    Theta1R = Theta1(indexI,:);
    regularizationTotal = regularizationTotal + Theta1R*Theta1R' - Theta1R(1,1)*Theta1R(1,1);
endfor

for (indexI = 1:size(Theta2,1))
    Theta2R = Theta2(indexI,:);
    regularizationTotal = regularizationTotal + Theta2R*Theta2R' - Theta2R(1,1)*Theta2R(1,1);
endfor

regularizationTotal = regularizationTotal*lambda/2;

J = (cost + regularizationTotal)/m;



sumDeltaHidden = zeros(size(Theta2));     % 10*26
sumDeltaInput = zeros(size(Theta1));      % 25*401

for (indexI = 1:m)
    dXI = [ones(1,1) X(indexI,:)]; % 1*401
    a1 = sigmoid(Theta1*dXI');    % 25*401 x 401*1 = 25*1
    a1CI = [ones(1,1); a1];        % 26*1
    
    a2 = sigmoid(Theta2*a1CI);     % 10*1
    yOfI = yVector(indexI,:)';

    deltaOutput = a2 - yOfI;              % 10*1
    sumDeltaHidden = sumDeltaHidden + deltaOutput*a1CI';   % 10 * 26

    deltaHidden = (Theta2'*deltaOutput).*a1CI.*(1 - a1CI);   % 26*10 x 10*1 . 26*1 . 26*1 = 26*1
    deltaHiddenWithoutC = deltaHidden(2:size(deltaHidden,1),:); % 25*1
    sumDeltaInput = sumDeltaInput + deltaHiddenWithoutC*dXI; % 25*401
endfor
Theta1S = Theta1(:,:);
Theta1S(:,1) = 0;
Theta2S = Theta2(:,:);
Theta2S(:,1) = 0;
 
Theta1_grad = (sumDeltaInput  + lambda*Theta1S)/m;
Theta2_grad = (sumDeltaHidden + lambda*Theta2S)/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
