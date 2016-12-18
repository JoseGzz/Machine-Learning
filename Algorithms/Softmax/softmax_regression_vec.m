function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);
 
  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  % theta=[theta,zeros(size(theta,1),1)];
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
 
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
 
%----------------------------
%{
%%% first try %%%
 
k = (1:10);
size(k); % 1 x 10
size(y); % 1 x 600
div = [ones(1,9); (log(exp(theta' * X) / exp(theta'*X)))];
%     [ (1x9) ; (9 x 60000) / (9 x 60000) ]     = (10 x 9)      
f = f + sum(bsxfun(@eq,k', y)' * div);
%   (60000 x 10) * (10 x 9) -> (60000 x 9)             
f = (-1)*sum(f);
f
%}
%------------------------------
 
%%% CALCULATE COST %%%
% get product of parameters and example values
 
product = [exp(theta' * X) ; ones(1,m)]; % (10 x 60,000)
 
% ignore last row by filling it with zeros
% product(10, :) = 0; 
% calculate log from the softmax formula
product_sum = sum(product, 1);
p = bsxfun(@rdivide, product, product_sum); % (10 x 60000)
A = p';
i = sub2ind(size(A), 1:size(A,1), y);
% use just the indexes which are equal to the y vector values,
% this is equivalent to 1{y_i = k} the indicator function
%i = sub2ind(size(log_p), y, 1:size(log_p,2));
% sum the results from just those indexes
f = f - sum(log(p(i)));
%f
 
%%% CALCULATE GRADIENT %%%
%{
ind_func = zeros(m,num_classes);
%ind_func = zeros(size(log_p'));
%size(ind_func)
%size(log_p')
ind_func(i) = 1;
g = -(X * (ind_func - log_p')); 
%g(:,end)=[];
g = g(1:end, 1:end-1);
%}
%{
  V= [exp(theta' * X) ; ones(1,m)];
  V_colsum= sum(V,1);
  h= bsxfun (@rdivide, V, V_colsum); % devide V by it's columnsum
 
  % Suppose we have a matrix A and we want to extract a single element from each row, where the column of the element to be extracted from row i is storedA in y(i), where y is a row vector.
  A=h';
  ind=sub2ind(size(A), 1:size(A,1), y);
  h_ind = A(ind)';
  f= -sum(log(h_ind));
 
  ind_y = zeros(m,num_classes);
  ind_y(ind)=1;
  ind_y= ind_y(:,1:(num_classes-1));
  h= h(1:(num_classes-1),:);
  g= -X* (ind_y-h');
  %}  
%{
  ind_y = zeros(m,num_classes);
  ind_y(i)=1;
  ind_y= ind_y(:,1:(num_classes-1));
  p = p(1:(num_classes-1),:);
  g= -X* (ind_y-p');
%}
product = [exp(theta' * X) ; ones(1,m)];
p = bsxfun (@rdivide, product, sum(product,1));
A = p';
ind = sub2ind(size(A), 1:size(A,1), y);
h_ind = A(ind)';
f = -sum(log(h_ind));
 
ind_y = zeros(m,num_classes);
ind_y(ind) = 1;
g = -X * (ind_y - p');
g = g(1:end, 1:end - 1);
g=g(:); % make gradient a vector for minFunc

