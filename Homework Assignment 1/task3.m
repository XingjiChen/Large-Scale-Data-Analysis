clear all;
clc;
tic;

% Parameters
n = 100; % Dimension of x
A = randn(n); 
A = A'*A; % Making A positive semi-definite
b = randn(n, 1);
x0 = eye(n, 1); % Initial guess
L = max(eig(A)); % Lipschitz constant
maxIt = 5000; % Maximum number of iterations
tol = 1e-5; % Convergence tolerance

% FISTA initialization
y = x0;
t = 1;
x = x0;
expConv = zeros(maxIt, 1); % To store objective function values
theoConv = zeros(maxIt, 1); % For storing theoretical convergence rate

% Initial objective value for theoretical rate calculation
fInitial = 0.5 * x0' * A * x0 - b' * x0;

% FISTA main loop
for k = 1:maxIt
    xOld = x;
    
    % Gradient step
    grad = A*y - b;
    x = y - (1/L) * grad;
    
    % Update t and y for the next iteration
    tOld = t;
    t = (1 + sqrt(1 + 4*t^2)) / 2;
    y = x + ((tOld - 1) / t) * (x - xOld);
    
    % Record the history (objective function value)
    expConv(k) = 0.5 * x' * A * x - b' * x;
    
    % Theoretical convergence rate
    theoConv(k) = fInitial * (1 / (k^2));
    
    % Check convergence
    if norm(A*x - b) <= tol
        expConv = expConv(1:k); % Truncate history to the current iteration
        theoConv = theoConv(1:k); % Truncate theoretical rate to the current iteration
        break;
    end
end

% Plot the convergence
figure;
plot(expConv, 'b', 'LineWidth', 1.5);
hold on;
theoConv = log(theoConv);
plot(theoConv, 'r--', 'LineWidth', 1.5);
hold off; % Make sure to turn hold off after plotting
title('FISTA Convergence');
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
legend('Experimental', 'Theoretical');
grid on;
toc;