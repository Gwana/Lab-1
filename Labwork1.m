
%  PART 1: Perceptron Classifier
%  Create a single layer perceptron classifier
%  y = 1  if  x1*w1 + x2*w2 + b > 0
%  y = -1 if  x1*w1 + x2*w2 + b <= 0

clear; clc;

% Random initialization of parameters
w1 = randn(1);
w2 = randn(1);
b  = randn(1);

% Learning rate controls how much weight changes after each error
delta = 0.1;

% training datasets perceptron learns from 4 samples(x1, x2 as features;)

X = [1 2;  % sample 1:  x1=1, x2=2
     2 1;  % ... 2
     -1 -2; % ..3
     -2 -1]; % ...4

d = [1; 1; -1; -1];  % Desired outputs for sample

% Number of training iterations
epochs = 20; % Times Perceptron goes through entire datasset


%  PART 2: Perceptron Training Algorithm
%  Updating:
%  w1 = w1 + eta * e(n) * x1(n)
%  w2 = w2 + eta * e(n) * x2(n)
%  b  = b  + eta * e(n)


for epoch = 1:epochs % repeats the training several times.
    for n = 1:size(X,1)
        % takes feautures of current training e.g
        x1 = X(n,1);
        x2 = X(n,2);

        % Perceptron output
        y = 1;
        if (x1*w1 + x2*w2 + b) <= 0
            y = -1;
        end

   % calculates the error the perceptron made for the current training e.g

        e = d(n) - y;

        % Update parameters
        w1 = w1 + delta * e * x1;
        w2 = w2 + delta * e * x2;
        b  = b  + delta * e;
    end
end

%% Display final learned parameters
disp('Final parameters:')
fprintf('w1 = %.3f\n', w1);
fprintf('w2 = %.3f\n', w2);
fprintf('b  = %.3f\n', b);
