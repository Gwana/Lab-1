

clc
clear all
close all

%% Function definition
x = 0.1:1/22:1;      % input value
d = (1 + 0.6 * sin(2*pi*x/0.7) + 0.3 * sin(2*pi*x)) / 2;  % desired output

% RBF parameters in first layer
c1 = 0.2;        % center of 1st RBF
r1 = 0.15;       % radius of 1st RBF
c2 = 0.9;        % center of 2nd RBF
r2 = 0.15;       % radius of 2nd RBF

% Output layer parameters second layer
w1 = rand(1);   
w2 = rand(1);    
b  = rand(1);    
delta = 0.1;       % learning rate

%% RBF training algorithm of RBF
for j = 1:400
    for i = 1:length(x)
        % RBF activations for current training sample x(i)
        F1(i) = exp(-(x(i) - c1).^2 ./ (2*r1^2));
        F2(i) = exp(-(x(i) - c2).^2 ./ (2*r2^2));  
        % output for sample i  (SCALARS!)
        y = w1 * F1(i) + w2 * F2(i) + b;
        % error for sample i
        e = d(i) - y;
        % gradient descent update
        w1 = w1 + delta * e * F1(i);  
        w2 = w2 + delta * e * F2(i);  
        b  = b  + delta * e;
    end
end
%% testing using more samples upto 50
x_test = 0.1:1/50:1;  
d_test = (1 + 0.6 * sin(2*pi*x_test/0.7) + 0.3 * sin(2*pi*x_test)) / 2;
y_test = zeros(1, length(x_test));
F1_test = zeros(1, length(x_test));
F2_test = zeros(1, length(x_test));
for n = 1:length(x_test)
    F1_test(n) = exp(-(x_test(n) - c1).^2 ./ (2*r1^2));
    F2_test(n) = exp(-(x_test(n) - c2).^2 ./ (2*r2^2));
    % output for test sample n (SCALARS!)
    y_test(n) = w1 * F1_test(n) + w2 * F2_test(n) + b;
end

% plots
figure(1)
plot(x_test, d_test, 'r-', 'LineWidth', 1.5); hold on
plot(x_test, y_test, 'b--', 'LineWidth', 1.5);
plot(x, d, 'ko', 'MarkerFaceColor', 'k');   % training points
legend('Target function', 'RBF approximation', 'Training samples');
xlabel('x'); ylabel('y');
title('RBF Network Approximation with 2 Gaussian Basis Functions');
grid on
figure(2)
plot(x, F1, 'b-o', x, F2, 'm-s', 'LineWidth', 1.2);
legend('F_1(x)', 'F_2(x)');
xlabel('x'); ylabel('RBF value');
title('Radial Basis Functions (Gaussians)');
grid on

% Displaying our learning parameters
disp('Learned output layer parameters (2nd layer):');
fprintf('w1 = %.4f\n', w1);
fprintf('w2 = %.4f\n', w2);
fprintf('b  = %.4f\n', b);