clear all;
clc;
tic;

n = 40000;
A = randn(n, n);
B = A^(-1);

toc;