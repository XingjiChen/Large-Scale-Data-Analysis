clear all;
clc;

% Dimension of the problem
n = 100; % matrix dimension
convergence = 1e-5; % convergence
max_iter = 5000; % maximum total number of iterations

% Generate positive semi-definite matrix A
A = randn(n, n);
A= A' * A;

% Generate vector b in the range of A
b = A * randn(n, 1);

% Initialization
x = zeros(n, 1);
y = x;
t = 1;
alpha = 1 / max(eig(A));% step size

grad_norm = norm(A * x - b);
% Perform FISTA iterations
iter = 0;
q_vals = [];
while(grad_norm > convergence && iter <= max_iter)    

    x_new = y - alpha  * (A*y - b);
    t_new = 1/2 * (1 + sqrt(1 + 4*t^2));
    y = x_new + ((t - 1) / t_new) * (x_new - x);
    
    % Compute quadratic function value
    quadratic_val = 1/2 * x_new' * A * x_new - b' * x_new;
    q_vals = [q_vals; quadratic_val];

    grad_norm = norm(A * x_new - b);

    x = x_new;
    t = t_new;
    iter = iter + 1;
end

x_opt = A \ b;
q_opt = 0.5 * x_opt' * A * x_opt - b' * x_opt;
% Experimental convergence rate
converge_experimental = log(abs(q_vals - q_opt));

iters = 1:iter;
converge_theoretical = log((2/alpha * norm(zeros(n, 1) - x_opt).^2)./(iters+1).^2);

% Plot the convergence rates
figure;
plot(iters, converge_experimental, 'b-', 'LineWidth', 2);
hold on;
plot(iters, converge_theoretical, 'r--', 'LineWidth', 2);
hold off;
xlabel('Iteration count');
ylabel('log(|f(x^t) - f^*|)');
title('Convergence Plot of FISTA Algorithm');
legend('Experimental', 'Theoretical','Location','north');
grid on;