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

vecCosto = zeros(1, m);
X = [ones(m,1) X];

#fprintf('Theta1: \n')
#size(Theta1')
#fprintf('Theta2: \n')
#size(Theta2')
#fprintf('sig: \n')
#size(sigmoid(X(1,:)))
%fprintf("input layer");
%input_layer_size

for i = 1:m
  y_class = zeros(1, num_labels);
  y_class(y(i)) = 1;
  vecCosto(i) = sum(-1 * y_class .* log(sigmoid([1 sigmoid(X(i,:) * Theta1')] * Theta2')) ... 
    - (1 - y_class) .* log(1 - sigmoid([1 sigmoid(X(i,:) * Theta1')] * Theta2')));
end

J = 1 / m * sum(vecCosto);
J = J + (lambda/(2*m)) * (sum(sumsq(Theta1(:, 2:input_layer_size + 1))) + sum(sumsq(Theta2(:, 2:hidden_layer_size + 1))));

% -------------------------------------------------------------

deltaM1 = zeros(size(Theta1));
deltaM2 = zeros(size(Theta2));

for t = 1:m
	a1 = X(t,:); % (1 x 401) ejemplo actual
	z2 = a1 * Theta1';  % (1 x 401) * (401 x 25) = (1 x 25)
	a2 = [1 sigmoid(z2)]; % (1 x 26)
	z3 = a2 * Theta2';  % (1 x 26) * (26 x 10) = (1 x 10)
	a3 = sigmoid(z3); % (1 x 10)

	yk = zeros(1, num_labels);  % (1 x 10)
	yk(y(t)) = 1; 
  
	delta3 = a3 - yk; % (1 x 10)
	delta2 = delta3 * Theta2 .* sigmoidGradient([1 z2]);  % (1 x 10) * (10 x 26) -> (1 x 26) .* (1 x 26)
	 
  	delta2 = delta2(2:end);
	deltaM1 = deltaM1 + delta2' * a1; % (26 x 1) * (1 x 401) -> (1 x 401)
	deltaM2 = deltaM2 + delta3' * a2; % (10 x 1) * (1 x 26) -> (10 x 26)
end

Theta1_grad = (1/m) * deltaM1;
Theta2_grad = (1/m) * deltaM2;

% -------------------------------------------------------------

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end); 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end); 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
