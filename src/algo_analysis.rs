use nalgebra::{DMatrix, DVector, Vector};
use std::time::{Duration, Instant};

use crate::algorithms;
use crate::csv_handler;



fn measure_avg_execution_time<F>(f: F) -> Duration
where
    F: Fn() -> (),
{
    /**
    Measures the average execution time of a function.

    Args:
    - f (F): Function to be measured.

    Returns:
    - Duration: Average execution time.
    */

    let num_iterations = 2; // (Set to 10 for replicating our experiments) Change here to modify the number of iterations to average the results.
    let mut total_duration = Duration::from_secs(0);

    for iter in 0..num_iterations {
        let start_time = Instant::now();
        f();
        let end_time = Instant::now();
        total_duration += end_time - start_time;
    }
    let averaged_duration = total_duration / num_iterations as u32;
    return averaged_duration
}

pub fn compute_time(A: &DMatrix<f64>, A_H: &DMatrix<f64>, r0: DVector<f64>, k_max: usize){
    /**
    Computes the execution time for various algorithms and writes the results to CSV files.

    Args:
    - A (&DMatrix<f64>): Input matrix A.
    - A_H (&DMatrix<f64>): Hermitian of input matrix A.
    - r0 (DVector<f64>): Initial vector.
    - k_max (usize): Maximum number of iterations.

    Returns:
    - None
    */

    let mut time_gs_vec: Vec<f64> = Vec::new();
    let mut time_cgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_H_vec: Vec<f64> = Vec::new();
    let mut time_lanczos_vec: Vec<f64> = Vec::new();
    

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    for k in &k_vector{
        println!("Iteration {:?}/{:?}", k, k_max);
        let averaged_time_gs = measure_avg_execution_time(|| {algorithms::gs(&A, r0.clone(), *k);});
        time_gs_vec.push(averaged_time_gs.as_secs_f64());

        let averaged_time_cgs = measure_avg_execution_time(|| {algorithms::arnoldi_cgs(&A, r0.clone(), *k);});
        time_cgs_vec.push(averaged_time_cgs.as_secs_f64());

        let averaged_time_mgs = measure_avg_execution_time(|| {algorithms::arnoldi_mgs(&A, r0.clone(), *k);});
        time_mgs_vec.push(averaged_time_mgs.as_secs_f64());

        let averaged_time_mgs_H = measure_avg_execution_time(|| {algorithms::arnoldi_mgs(&A_H, r0.clone(), *k);});
        time_mgs_H_vec.push(averaged_time_mgs_H.as_secs_f64());

        let averaged_time_lanczos = measure_avg_execution_time(|| {algorithms::lanczos(&A_H, r0.clone(), *k);});
        time_lanczos_vec.push(averaged_time_lanczos.as_secs_f64());
    }

    csv_handler::write_vector_to_csv(&time_gs_vec, "./experiment_results/gs/time_gs_vec.csv");
    csv_handler::write_vector_to_csv(&time_cgs_vec, "./experiment_results/cgs/time_cgs_vec.csv");
    csv_handler::write_vector_to_csv(&time_mgs_vec, "./experiment_results/mgs/time_mgs_vec.csv");
    csv_handler::write_vector_to_csv(&time_mgs_H_vec, "./experiment_results/mgs/time_mgs_H_vec.csv");
    csv_handler::write_vector_to_csv(&time_lanczos_vec, "./experiment_results/lanczos/time_lanczos_vec.csv");

}

pub fn are_columns_orthonormal(matrix: &DMatrix<f64>) -> bool {
    /**
    Checks if the columns of a matrix are orthonormal.

    Args:
    - matrix (&DMatrix<f64>): Input matrix.

    Returns:
    - bool: Whether the columns are orthonormal or not.
    */

    let num_columns = matrix.ncols();

    for i in 0..num_columns {
        let col_i = matrix.column(i);

        // Check if the magnitude is close to 1 (considering floating-point precision)
        let magnitude_i = Vector::norm(&col_i);
        if (magnitude_i - 1.0).abs() > 1e-10 {
            return false;
        }

        for j in 0..i {
            let col_j = matrix.column(j);

            // Check if the dot product is close to zero (considering floating-point precision)
            let dot_product = col_i.dot(&col_j);
            if (dot_product - if i == j { 1.0 } else { 0.0 }).abs() > 1e-10 {
                return false;
            }
        }
    }

    true
}

pub fn orthogonality_loss(V: &DMatrix<f64>) -> f64 {
    /**
    Calculates the orthogonality loss of a matrix.

    Args:
    - V (&DMatrix<f64>): Input matrix.

    Returns:
    - f64: Orthogonality loss.
    */

    let T = V.transpose()*V;
    let I = DMatrix::<f64>::identity(T.nrows(), T.ncols());
    return (I-T).norm()
}