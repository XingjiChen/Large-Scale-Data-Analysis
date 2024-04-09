%% Taks 2: Conjugate Gradient Algorithm
n = 100; % matrix dimension
convergence = 1e-5; % convergence
max_iter = 5000; % maximum total number of iterations

A = randn(n, n);
A= A * A';
% Generate vector b in the range of A
b = A * randn(n, 1);

x = zeros(n, 1); % Initialize x
q_0 = 0.5 * x' * A * x - b' * x;
residual0 = A * x - b; % Initial residual
p = -residual0; % Initial search direction
residual_norm =  norm(residual0);

% Compute optimal x using matrix inversion for comparison
x_opt = A \ b;

% Conjugate Gradient Algorithm
iter = 0;
grad_norm = norm(A * x - b);
q_vals = [];
norm_ei_A_vector = [];
while(grad_norm > convergence && iter <= max_iter)
    % step size
    alpha = (residual0' * residual0) / (p' * A * p);
    x = x + alpha * p;
    residual_new = residual0 + alpha * A * p;

    % Compute quadratic function value
    quadratic_val = 1/2 * x' * A * x - b' * x;
    q_vals = [q_vals; quadratic_val];

    beta = (residual_new' * residual_new) / (residual0' * residual0);
    p = -residual_new + beta * p;
    residual0 = residual_new;
    
    grad_norm = norm(A * x - b);
    e = x - x_opt;
    norm_ei_A = sqrt(e' * A * e);
    norm_ei_A_vector = [norm_ei_A_vector;norm_ei_A];
    iter = iter + 1;
end

converge_experimental = log(norm_ei_A_vector);

k = cond(A);
e0 = zeros(n, 1) - x_opt;
norm_e0_A = sqrt(e0' * A * e0);
converge_theoretical = 2 * ((sqrt(k) - 1) / (sqrt(k) + 1)).^(1:iter) * norm_e0_A;

% Plot the experimental convergence rate
iters = 1:iter;
plot(iters, converge_experimental, 'b', 'LineWidth', 2);
hold on;
plot(iters, log(converge_theoretical), 'r--', 'LineWidth', 2);
hold off;
xlabel('Iteration count');
ylabel('log||x^t - x^*||)');
title('Convergence Plot of Conjugate Gradient Algorithm');
legend('Experimental', 'Theoretical','Location','south');
grid on;