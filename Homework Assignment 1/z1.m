%% Task 1: Gradient Descent Algorithm
n = 100; % matrix dimension
convergence = 1e-5; % convergence
max_iter = 5000; % maximum total number of iterations

% Generate positive semi-definite matrix A
A = randn(n, n);
A = A * A';

mu = min(eigenvalues); % A 的最小特征值
L = max(eigenvalues); % A 的最大特征值
alpha = 1 / L; % 步长，确保步长小于 1/L

% Generate vector b in the range of A
b = A * randn(n, 1);

% Initialize x
x = zeros(n, 1);

grad_norm = norm(A * x - b);
iter = 0;
q_vals = [];
grad_norms_vector = [];
% && iter < max_iter
while(grad_norm > convergence && iter < max_iter)
    % Gradient descent update
    grad = A * x - b;
    x = x - alpha * grad;
    
    % Compute quadratic function value
    quadratic_val = 1/2 * x' * A * x - b' * x;
    q_vals = [q_vals; quadratic_val];
    
    % Compute norm of the gradient
    grad_norm = norm(grad);
    grad_norms_vector = [grad_norms_vector; grad_norm];
    iter = iter + 1;
end

% Compute optimal x using matrix inversion for comparison
x_opt = A \ b;
q_opt = 0.5 * x_opt' * A * x_opt - b' * x_opt;

% Plot the convergence plot
iter_vector = 1:iter;
converge_experimental = log(q_vals - q_opt);
%converge_theoretical = log(grad_norms_vector * norm(zeros(n, 1) - x_opt));
%converge_theoretical = log(2 / alpha * norm(zeros(n, 1) - x_opt) ./ iter_vector);

figure;
hold on
plot(iter_vector, converge_experimental, 'b', 'LineWidth', 2);
plot(iter_vector, converge_theoretical, 'r--', 'LineWidth', 2);
legend('Experimental','Theoretical');
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
title('Convergence Plot of Gradient Descent');
grid on