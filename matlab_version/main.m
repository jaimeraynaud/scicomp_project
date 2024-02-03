clc; clear; close all;
format compact
n = 250;
k = 200;

%A = diag(1:n);
A = randn(n,n) + 1i*randn(n,n);
v = randn(length(A),1);

compute_time(A, A+A', v, k);
disp('Done!');

function averaged_duration = measure_avg_execution_time(f)
    num_iterations = 10; % Change here to modify the number of iterations to average the results.
    
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
        averaged_time_gs = measure_avg_execution_time(@( ) qr_krylov(A, v, k));
        time_gs_vec = [time_gs_vec, averaged_time_gs];

        averaged_time_cgs = measure_avg_execution_time(@( ) cgsarnoldi(A, v, k));
        time_cgs_vec = [time_cgs_vec, averaged_time_cgs];

        averaged_time_mgs = measure_avg_execution_time(@( ) mgsarnoldi(A, v, k));
        time_mgs_vec = [time_mgs_vec, averaged_time_mgs];

        averaged_time_mgs_H = measure_avg_execution_time(@( ) mgsarnoldi(A_H, v, k));
        time_mgs_H_vec = [time_mgs_H_vec, averaged_time_mgs_H];

        averaged_time_lanczos = measure_avg_execution_time(@( ) lanczos(A_H, v, k));
        time_lanczos_vec = [time_lanczos_vec, averaged_time_lanczos];
    end
    
    csvwrite('../python_visualizations/gs/time_gs_vec_matlab.csv', time_gs_vec);
    csvwrite('../python_visualizations/cgs/time_cgs_vec_matlab.csv', time_cgs_vec);
    csvwrite('../python_visualizations/mgs/time_mgs_vec_matlab.csv', time_mgs_vec);
    csvwrite('../python_visualizations/mgs/time_mgs_H_vec_matlab.csv', time_mgs_H_vec);
    csvwrite('../python_visualizations/lanczos/time_lanczos_vec_matlab.csv', time_lanczos_vec);
    
end
 

% GramSchmidt for orthonormal Krylov subspace basis
function [V,H]=qr_krylov(A,v,k)
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

% Arnoldi algorithm - classical GramSchmidt variant (Algorithm 5)
function [V,H]=cgsarnoldi(A,v,k)
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

function [V,H]=mgsarnoldi(A,v,k)
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

% Lanczos method (Algorithm 7)
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

