%% Coordinate Descent Algorithm
n = 100; % matrix dimension
convergence = 1e-5; % convergence
max_iter = 5000; % maximum total number of iterations

% Generate positive semi-definite matrix A
A = randn(n, n);
A= A' * A;

b = A * randn(n, 1);

x = zeros(n, 1); % Initial x
alpha = 1 / max(eig(A));% step size

q_vals = [];
grad_norm = norm(A * x - b);
iter = 0;
while(grad_norm > convergence && iter <= max_iter)
    for i = 1:n
        x(i) = (b(i) - A(i, [1:i-1, i+1:n]) * x([1:i-1, i+1:n])) / A(i, i);
    end
    % Compute quadratic function value
    quadratic_val = 1/2 * x' * A * x - b' * x;
    q_vals = [q_vals; quadratic_val];

    grad_norm = norm(A * x - b);
    iter = iter + 1;
end

x_opt = A \ b;
q_opt = 0.5 * x_opt' * A * x_opt - b' * x_opt;
% Experimental convergence rate
converge_experimental = log(abs(q_vals - q_opt));
iters = 1:iter;
converge_theoretical = log((1/alpha * norm(zeros(n, 1) - x_opt)/2./iters));


% Plot the convergence
figure;
plot(iters,converge_experimental, 'b', 'LineWidth', 2);
hold on;
plot(iters,converge_theoretical, 'r--', 'LineWidth', 2);
xlabel('Iteration');
ylabel('log(|f(x^t) - f^*|)');
title('Convergence of Coordinate Descent Algorithm');
legend('Experimental Convergence', 'Theoretical Convergence');
grid on;