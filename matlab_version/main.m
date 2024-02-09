clc; clear; close all;
format compact

% MATLAB implementation analogous to our Rust implementation.

disp("===========================================================================================================================");
disp("Comparison of algorithms for computing orthonormal bases");
disp("===========================================================================================================================", newline);

n = 75; % (Set to 250 for replicating our experiments) Size of the initial matrix A (n > k)
k_max = 70; % (Set to 200 for replicating our experiments) (n > k)

disp("===========================================================================================================================");
disp(['Initial values: n (size of the matrix) = ', num2str(n), ', k (dimension of the subspace) = ', num2str(k_max)]);
disp("===========================================================================================================================", newline);

% Reading a CSV file containing the matrix A
A = readmatrix('../experiment_results/A.csv');
A = A(:, 1:end-1); % (Uncomment for older versions of MATLAB)

A_H = readmatrix('../experiment_results/A_H.csv');
A_H = A_H(:, 1:end-1); % (Uncomment for older versions of MATLAB)

v = readmatrix('../experiment_results/r0.csv');

disp("===========================================================================================================================");
disp("Computing average runtime...");
compute_time(A, A_H, v, k_max);
disp("===========================================================================================================================", newline);
disp("Finished");

function averaged_duration = measure_avg_execution_time(f)
    num_iterations = 2; % (Set to 10 for replicating our experiments) Change here to modify the number of iterations to average the results.
    
    total_duration = 0;

    for i = 1:num_iterations
        duration = seconds(timeit(f));
        total_duration = total_duration + duration;
        %disp(total_duration);
    end

    averaged_duration = seconds(total_duration) / num_iterations;
    
end

function compute_time(A, A_H, v, k_max)
    time_gs_vec = [];
    time_cgs_vec = [];
    time_mgs_vec = [];
    time_mgs_H_vec = [];
    time_lanczos_vec = [];

    k_vector = 2:k_max;

    for k = k_vector
        disp(['Iteration: ', num2str(k), '/', num2str(k_max)]);
        averaged_time_gs = measure_avg_execution_time(@( ) gs(A, v, k));
        time_gs_vec = [time_gs_vec, averaged_time_gs];

        averaged_time_cgs = measure_avg_execution_time(@( ) arnoldi_cgs(A, v, k));
        time_cgs_vec = [time_cgs_vec, averaged_time_cgs];

        averaged_time_mgs = measure_avg_execution_time(@( ) arnoldi_mgs(A, v, k));
        time_mgs_vec = [time_mgs_vec, averaged_time_mgs];

        averaged_time_mgs_H = measure_avg_execution_time(@( ) arnoldi_mgs(A_H, v, k));
        time_mgs_H_vec = [time_mgs_H_vec, averaged_time_mgs_H];

        averaged_time_lanczos = measure_avg_execution_time(@( ) lanczos(A_H, v, k));
        time_lanczos_vec = [time_lanczos_vec, averaged_time_lanczos];
    end
    
    csvwrite('../experiment_results/gs/time_gs_vec_matlab.csv', time_gs_vec);
    csvwrite('../experiment_results/cgs/time_cgs_vec_matlab.csv', time_cgs_vec);
    csvwrite('../experiment_results/mgs/time_mgs_vec_matlab.csv', time_mgs_vec);
    csvwrite('../experiment_results/mgs/time_mgs_H_vec_matlab.csv', time_mgs_H_vec);
    csvwrite('../experiment_results/lanczos/time_lanczos_vec_matlab.csv', time_lanczos_vec);
    
end
 

% Gram-Schmidt for orthonormal Krylov subspace basis
function [V,H]=gs(A,v,k)
n = length(A);  
V = zeros(n,k+1); 
V(:,1) = v/norm(v);
H = zeros(k+1,k);
for j=1:k
    v = A*v;
    vtilde = v;
    for i=1:j
        H(i,j) = V(:,i)'*v;
        vtilde = vtilde - H(i,j)*V(:,i);
    end
    H(j+1,j) = norm(vtilde);
	V(:,j+1) = vtilde/H(j+1,j);
end
end

% Arnoldi algorithm - classical Gram-Schmidt variant
function [V,H]=arnoldi_cgs(A,v,k)
n = length(A); 
V = zeros(n,k+1); 
V(:,1) = v/norm(v);
H = zeros(k+1,k);
for j=1:k
    Av = A*V(:,j);
    vtilde = Av;
    for i=1:j
        H(i,j) = V(:,i)'*Av;
        vtilde = vtilde - H(i,j)*V(:,i);
    end
    H(j+1,j) = norm(vtilde);
	V(:,j+1) = vtilde/H(j+1,j);
end
end

function [V,H]=arnoldi_mgs(A,v,k)
n = length(A); 
V = zeros(n,k+1); 
V(:,1) = v/norm(v);
H = zeros(k+1,k);
for j=1:k
	V(:,j+1) = A*V(:,j);
    for i=1:j
		H(i,j) = V(:,i)'*V(:,j+1);
		V(:,j+1) = V(:,j+1)-H(i,j)*V(:,i);
    end
    for i=1:j
		H(i,j) = H(i,j) + V(:,i)'*V(:,j+1);
		V(:,j+1) = V(:,j+1)-(V(:,i)'*V(:,j+1))*V(:,i);
    end
	H(j+1,j) = norm(V(:,j+1));
	V(:,j+1) = V(:,j+1)/H(j+1,j);
end
end

% Lanczos algorithm
function [V,T] = lanczos(A,v,k)
V = zeros(length(A),k+1); V(:,1) = v/norm(v,2);
T = zeros(k+1,k);
for j = 1:k
    if j ~= 1
        u = A*V(:,j) - T(j,j-1)*V(:,j-1);
    else
        u = A*V(:,j); 
    end
    T(j,j) = V(:,j)'*u;
    vhat = u - T(j,j)*V(:,j);
    T(j+1,j) = norm(vhat,2);
    if j ~= k
        T(j,j+1) = T(j+1,j);
    end
    V(:,j+1) = vhat/T(j+1,j);
end
end

