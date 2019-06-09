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

% Add ones to the X data matrix
X = [ones(m, 1) X];

a1=X;

z1=Theta1*a1';

a2=sigmoid(z1);
a2=a2';

% add 1s to a2
a2 = [ones(size(a1,1), 1) a2];
z2=a2*Theta2';
H=sigmoid(z2);

% change y to 0 and 1 format. It is 5000X1. we need to change it to 5000X10.
% One row for each label with 10 col(0s and 1s).
I = eye(num_labels);
newY=zeros(m, num_labels);
for c=1:m
 newY(c, :)= I(y(c), :);
endfor

%we need to sum each row first and then take sum of each row unlike direct sum. It means convert 5000X10
% matrix to 5000X1 matrix first and then do a sum again. The argument 2 does the same thing
% calculate J

% calculate regularization term for part2

newTheta1=Theta1;
 
newTheta1(:,1)=0;

newTheta2=Theta2;

newTheta2(:,1)=0;

R=sum(((lambda/(2*m))*sum(newTheta1.^2)))+sum(((lambda/(2*m))*sum(newTheta2.^2)));

J = sum(sum((-newY).*log(H) - (1-newY).*log(1-H), 2))/m+R;


%backpropagation
delta3=H-newY;
Theta2_grad=(delta3'*a2)/m+(lambda/(m))*newTheta2;
gPrimeZ3=a2.*(1-a2);
delta2=((Theta2(:,2:end))'*delta3')'.*gPrimeZ3(:,2:end);
Theta1_grad=(delta2'*a1)/m+(lambda/(m))*newTheta1;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
