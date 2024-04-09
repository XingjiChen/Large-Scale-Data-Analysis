clear all;
clc;
tic;

% Initialize parameters
n = 100; % Dimension of x and A
A = randn(n, n);
A = A'*A; % Ensuring A is symmetric positive definite
b = randn(n, 1); % Random vector b

% Conjugate Gradient (CG) Algorithm setup
x = zeros(n, 1); % Initial guess for the solution
r = b - A*x; % Initial residual
p = r; % Initial direction
rsOld = r'*r; % Initial squared residual norm
tol = 1e-5; % Tolerance for convergence
maxIt = 5000; % Maximum number of iterations
expConv = []; % To store objective values for plotting

% Theoretical convergence rate setup
eigVals = eig(A);
minEig = min(eigVals);
maxEig = max(eigVals);
alpha = 1 / maxEig;
kappa = maxEig / minEig; % Condition number of A
theoConv = []; % To store theoretical convergence rates

% CG Algorithm main loop
for i = 1:maxIt
    Ap = A * p;
    alpha = rsOld / (p'*Ap); % Step size
    x = x + alpha * p; % Update solution
    r = r - alpha * Ap; % Update residual
    rsNew = r'*r; % Update squared residual norm
    
    % Compute objective function value at x
    objVal = 0.5 * x' * A * x - b' * x;
    objVal = 1 / objVal;
    objVal = log(abs(objVal));
    expConv = [expConv; objVal]; % Store for plotting

    % Compute and store theoretical convergence rate
    if i == 1
        e0 = objVal; % Initial error for theoretical rate calculation
    end
    theoConv = [theoConv; log(abs(e0 * (2 * ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^i)))];
    
    if rsNew < tol % Check convergence
        break;
    end
    
    p = r + (rsNew/rsOld) * p; % Update direction
    rsOld = rsNew; % Update squared residual norm for next iteration
end

% Plotting convergence
figure;
plot(expConv, 'b', 'LineWidth', 1.5); % Experimental convergence
hold on;
plot(theoConv, '--', 'LineWidth', 1.5); % Theoretical convergence
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
title('Convergence of Conjugate Gradient Algorithm');
legend('Experimental', 'Theoretical');
grid on;
toc;