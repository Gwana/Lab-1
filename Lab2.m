clc
clear all
%% Defining our values
x=0.1:1/22:1;   % input values
d= (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x)) / 2;  % desired output
%% initialize weights
% first hidden layer
w11_1=rand(1);
w12_1=rand(1);
w13_1=rand(1);
w14_1=rand(1);
w15_1=rand(1);   
b1_1=rand(1);
b2_1=rand(1);
b3_1=rand(1);
b4_1=rand(1);
b5_1=rand(1);    
% hidden layer output -> output neuron weights
w11_2=rand(1);
w12_2=rand(1);
w13_2=rand(1);
w14_2=rand(1);
w15_2=rand(1);   
b1_2=rand(1);
delta = 0.1; % learning rate
for j=1:5000     
    for i=1:length(x)
        % feed-forward
        % weighted sums (input * weight + bias)
        v1_1 = x(i)*w11_1 + b1_1;
        v2_1 = x(i)*w12_1 + b2_1;
        v3_1 = x(i)*w13_1 + b3_1;
        v4_1 = x(i)*w14_1 + b4_1;
        v5_1 = x(i)*w15_1 + b5_1;   
        % Activation (tanh)
        y1_1 = tanh(v1_1);
        y2_1 = tanh(v2_1);
        y3_1 = tanh(v3_1);
        y4_1 = tanh(v4_1);
        y5_1 = tanh(v5_1);         
        % Output layer weighted sum (5 hidden neurons)
        v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + y4_1 * w14_2 + y5_1 * w15_2 + b1_2;
        % second layer activation (linear)
        y1_2 = v1_2;
        % final output
        y = y1_2;
        % Back propagation
        % error
        e = d(i) - y;

        % error gradient for output layer just 1 neuron, so deri = 1
        delta1_2 = e;  
        % error gradients for hidden layer neurons
        delta1_1 = (1 - tanh(v1_1)^2) * delta1_2 * w11_2;
        delta2_1 = (1 - tanh(v2_1)^2) * delta1_2 * w12_2;
        delta3_1 = (1 - tanh(v3_1)^2) * delta1_2 * w13_2;
        delta4_1 = (1 - tanh(v4_1)^2) * delta1_2 * w14_2;
        delta5_1 = (1 - tanh(v5_1)^2) * delta1_2 * w15_2;  
        % update weights - output layer
        w11_2 = w11_2 + delta * delta1_2 * y1_1;
        w12_2 = w12_2 + delta * delta1_2 * y2_1;
        w13_2 = w13_2 + delta * delta1_2 * y3_1;
        w14_2 = w14_2 + delta * delta1_2 * y4_1;
        w15_2 = w15_2 + delta * delta1_2 * y5_1;  
        b1_2  = b1_2  + delta * delta1_2;
        % hidden layer weights update
        w11_1 = w11_1 + delta * delta1_1 * x(i);
        w12_1 = w12_1 + delta * delta2_1 * x(i);
        w13_1 = w13_1 + delta * delta3_1 * x(i);
        w14_1 = w14_1 + delta * delta4_1 * x(i);
        w15_1 = w15_1 + delta * delta5_1 * x(i);  
        b1_1 = b1_1 + delta * delta1_1;
        b2_1 = b2_1 + delta * delta2_1;
        b3_1 = b3_1 + delta * delta3_1;
        b4_1 = b4_1 + delta * delta4_1;
        b5_1 = b5_1 + delta * delta5_1;          
    end
end
%% Testing network with new set of values
x_test = 0.1:1/50:1;
d_test = (1 + 0.6*sin(2*pi*x_test/0.7) + 0.3*sin(2*pi*x_test)) / 2;
y_test = zeros(1, length(x_test));
% Calculate response
for n=1:length(x_test)
    % feed-forward
    v1_1 = x_test(n)*w11_1 + b1_1;
    v2_1 = x_test(n)*w12_1 + b2_1;
    v3_1 = x_test(n)*w13_1 + b3_1;
    v4_1 = x_test(n)*w14_1 + b4_1;
    v5_1 = x_test(n)*w15_1 + b5_1;   
    % activation
    y1_1 = tanh(v1_1);
    y2_1 = tanh(v2_1);
    y3_1 = tanh(v3_1);
    y4_1 = tanh(v4_1);
    y5_1 = tanh(v5_1);              
    % output layer
    v1_2 = y1_1 * w11_2 + ...
           y2_1 * w12_2 + ...
           y3_1 * w13_2 + ...
           y4_1 * w14_2 + ...
           y5_1 * w15_2 + ...       
           b1_2;
    y1_2 = v1_2;
    y_test(n) = y1_2;
end
%% Plotting test results
figure(2)
plot(x, d, 'r--o', x_test, y_test, 'b-*')
legend('Original function', 'Training algorithm output')
title('Function approximation with 5 hidden neurons')