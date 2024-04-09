clear all;
clc;
tic;

n = 100; % Dimension of x as specified in the task
A = randn(n, n);
A = A' * A; % Make A positive semi-definite
b = A * randn(n, 1); % Generate a random vector b

eigVals = eig(A);
minEig = min(eigVals);
maxEig = max(eigVals);
alpha = 1 / maxEig;

% Coordinate Descent parameters
x = zeros(n, 1); % Initial guess
tol = 1e-5;
maxIt = 5000;
objVals = []; % To store objective values for plotting convergence
xOpt = A \ b;
yOpt = 0.5 * xOpt' * A * xOpt - b' * xOpt;

% Coordinate Descent Algorithm
for iter = 1:maxIt
    for i = 1:n
        x(i) = (b(i) - A(i, [1:i-1, i+1:n]) * x([1:i-1, i+1:n])) / A(i, i);
    end
    
    % Calculate objective value
    objVal = 0.5 * x' * A * x - b' * x;
    objVals = [objVals; objVal];

    % Check for convergence (this is a simple criterion, in practice, might need a more robust one)
    if iter > 1 && abs(log(abs(objVals(end) - yOpt))) < tol
        break;
    end
end

iters = 1:iter;
theoConv = log((1/alpha * norm(zeros(n, 1) - xOpt)/2./iters));
expConv = log(objVals - yOpt);

% Plot the convergence
figure;
plot(expConv, 'b', 'LineWidth', 1.5);
hold on;
plot(theoConv, 'r--', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
title('Convergence of Coordinate Descent Algorithm');
legend('Experimental', 'Theoretical');
grid on;
toc;