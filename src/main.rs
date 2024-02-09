#![allow(warnings)]

/**
Scientific Computing project: Comparison of Algorithms for Computing Orthonormal Bases

Samuel Aliprandi (492247)
Jaime Raynaud (492252)
Dagur Oskarsson (492256)

Technische Universität Berlin
*/

mod algorithms;
mod algo_analysis;
mod csv_handler;

use nalgebra::{DMatrix, DVector, Vector};


fn orchestrator() {
    /**
    Orchestrates the comparison of algorithms for computing orthonormal bases in Rust.

    This function performs the following steps:
    1. Initializes the matrices and vectors.
    2. Computes orthonormal bases using various algorithms.
    3. Calculates orthogonality loss for each algorithm.
    4. Writes orthogonality loss and time vectors to CSV files.

    Args:
    - None

    Returns:
    - None
    */

    println!("\n===========================================================================================================================");
    println!("Comparison of algorithms for computing orthonormal bases in Rust");
    println!("===========================================================================================================================\n");

    // Set the number of Arnoldi iterations
    let n = 75; // (Set to 250 for replicating our experiments) Size of the initial matrix A (n > k_max)
    let k_max = 70; // (Set to 200 for replicating our experiments) (n > k_max)


    println!("===========================================================================================================================");
    println!("Initial values: n (size of the matrix) = {:?}, k (dimension of the subspace) = {:?}", n, k_max);
    println!("===========================================================================================================================\n");

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    let A_H = A.clone()+A.transpose();

    let _ = csv_handler::write_matrix_to_csv(&A, &format!("./experiment_results/A.csv"));
    let _ = csv_handler::write_matrix_to_csv(&A_H, &format!("./experiment_results/A_H.csv"));

    // Choose a random initial vector and normalize it
    let mut r0 = DVector::new_random(n);
    r0.normalize_mut();

    let vec_r0: Vec<f64> = r0.iter().cloned().collect();
    let _ = csv_handler::write_vector_to_csv(&vec_r0, "./experiment_results/r0.csv");

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    let mut orthogonality_loss_gs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_cgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_H_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_lanczos_vec: Vec<f64> = Vec::new();

    println!("===========================================================================================================================");
    println!("Computing orthonormal bases with GS, CGS, MGS and Lanczos...");

    for k in &k_vector{
        println!("Iteration {:?}/{:?}", k, k_max);

        // GS iteration
        let (V_gs, R_gs) = algorithms::gs(&A, r0.clone(), *k);
        orthogonality_loss_gs_vec.push(algo_analysis::orthogonality_loss(&V_gs));

        // Arnoldi iteration Classical GS
        let (V_cgs, H_cgs) = algorithms::arnoldi_cgs(&A, r0.clone(), *k);
        orthogonality_loss_cgs_vec.push(algo_analysis::orthogonality_loss(&V_cgs));

        // Arnoldi iteration Modified GS
        let (V_mgs, H_mgs) = algorithms::arnoldi_mgs(&A, r0.clone(), *k);
        orthogonality_loss_mgs_vec.push(algo_analysis::orthogonality_loss(&V_mgs));

        // Arnoldi iteration Modified GS (Hermitian matrix)
        let (V_mgs_H, H_mgs_H) = algorithms::arnoldi_mgs(&A_H, r0.clone(), *k);
        orthogonality_loss_mgs_H_vec.push(algo_analysis::orthogonality_loss(&V_mgs_H));

        // Arnoldi iteration Modified GS
        let (V_lanczos, T_lanczos) = algorithms::lanczos(&A_H, r0.clone(), *k);
        orthogonality_loss_lanczos_vec.push(algo_analysis::orthogonality_loss(&V_lanczos));
        
        // Write orthogonality loss and time vectors to csv
        csv_handler::write_orthogonality_vectors_to_csv(&orthogonality_loss_gs_vec, &orthogonality_loss_cgs_vec, &orthogonality_loss_mgs_vec, &orthogonality_loss_mgs_H_vec, &orthogonality_loss_lanczos_vec);

        // Write matrices to CSV files
        csv_handler::write_matrices_to_csv(&V_gs, &R_gs, &V_cgs, &H_cgs, &V_mgs, &H_mgs, &V_lanczos, &T_lanczos, *k);

    }
    println!("===========================================================================================================================\n");

    // Compute and write averaged execution time.
    println!("===========================================================================================================================");
    println!("Computing average runtime...");
    algo_analysis::compute_time(&A, &A_H,r0, k_max);
    println!("===========================================================================================================================\n");

    println!("Finished\n")


}



fn main() {
    orchestrator()
}