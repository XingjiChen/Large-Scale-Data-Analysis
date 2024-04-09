clear all;
clc;
tic;

% Initialization
n = 100;
A = randn(n, n);
A = A * A'; % Ensuring A is symmetric positive definite
b = A * randn(n, 1);
x = zeros(n, 1);

% Calculating eigenvalues for step size and condition number
eigVals = eig(A);
minEig = min(eigVals);
maxEig = max(eigVals);
alpha = 1 / maxEig;
kappa = maxEig / minEig;

% Tolerance and max iterations setup
tol = 1e-5;
maxIt = 5000;

% Exact solution and its function value for convergence analysis
xOpt = A\b;
fOpt = 1/2 * xOpt' * A * xOpt - b' * xOpt;

% Gradient norm and iteration counter initialization
gradNorm = norm(A * x - b);
iter = 0;

% Arrays for function values and gradient norms
fVals = [];
gNorms = [];

% Gradient descent loop
while(gradNorm > tol && iter < maxIt)
    grad = A * x - b; % Gradient computation
    x = x - alpha * grad; % Solution update
    f = 1/2 * x' * A * x - b' * x; % Current function value
    fVals = [fVals; f];
    gradNorm = norm(grad); % Gradient norm update
    % gNorms = [gNorms; gradNorm];
    iter = iter + 1;
end

% Data preparation for plotting
iters = 1:iter;
expConv = log(fVals - fOpt);
theoConv = log(2 / alpha * norm(zeros(n, 1) - xOpt) ./ iters);

% Plotting convergence analysis
figure;
hold on;
plot(expConv, 'b', 'LineWidth', 1.5);
plot(theoConv, 'r--', 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
legend('Experimental', 'Theoretical');
title('Convergence of Gradient Descent Algorithm');
grid on;
hold off;
toc;