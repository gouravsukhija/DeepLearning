function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



Z=X*theta;

H=sigmoid(Z);

temp1=0-(y.*log(H));
temp2=(1-y);

temp3=temp2.*log(1-H);

J=sum(temp1-temp3)/m

% calculate lambda expression

 newTheta=theta;
 
 newTheta(1,1)=0;

 temp4=sum(newTheta.^2);


J=J+(lambda/2*m)*(temp4);


% if the theta is row vector then h is theta'*X
% if theta is column vector then h is theta*X

  

 
 grad = (1/m) * X' * (H - y);

 grad=grad+(lambda/m)*newTheta;



% =============================================================

end
