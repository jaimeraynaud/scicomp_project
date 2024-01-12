use std::ops::Div;

use nalgebra::{DMatrix, DVector, Vector, MatrixMN, U1, Dynamic, Vector1};
use nalgebra::linalg::QR;
use ndarray::{Array2, ArrayView1};


fn arnoldi_cg_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
    let mut H = DMatrix::zeros(n+1, n);
    let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(A.nrows(), n+1);


    V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

    for k in 1..n+1 {
        let mut v_ = A * &V.column(k - 1);  // Generate a new candidate vector
        for i in 0..k {
            H[(i, k - 1)] = (A * &V.column(k - 1)).conjugate().dot(&V.column(i));
            v_ = v_ - H[(i, k - 1)]*&V.column(i);
        }  // Subtract the projections on previous vectors
        H[(k, k-1)] = v_.norm();

        if H[(k, k-1)] > eps{
            V.set_column(k, &(v_ / H[(k, k-1)]));
        } else { // Add the produced vector to the list, unless 
            return (V, H);
        }  // If that happens, stop iterating.
    }
            
    return (V, H);
}

fn arnoldi_mg_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
    let mut H = DMatrix::zeros(n+1, n);
    let mut V = DMatrix::zeros(A.nrows(), n+1);


    V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

    for k in 1..n+1 {
        let mut v_ = A * &V.column(k - 1);  // Generate a new candidate vector
        for j in 0..k {
            H[(j, k - 1)] = V.column(j).conjugate().dot(&v_);
            v_ = v_ - H[(j, k - 1)] * V.column(j);
        }  // Subtract the projections on previous vectors
        H[(k, k-1)] = v_.norm();

        if H[(k, k-1)] > eps{
            V.set_column(k, &(v_ / H[(k, k-1)]));
        } else { // Add the produced vector to the list, unless 
            return (V, H);
        }  // If that happens, stop iterating.
    }
            
    return (V, H);
}

fn are_columns_orthonormal(matrix: &DMatrix<f64>) -> bool {
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

fn main() {
    // Set the number of Arnoldi iterations
    let n = 50; // The number of iterations give the number of orthonormal columns
    let k = 45; // Grade of the vector b

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    // Choose a random initial vector for Arnoldi algorithm
    let mut r0 = DVector::new_random(n);

    // Arnoldi iteration Classical GS
    let (V, H) = arnoldi_cg_iteration(&A, r0.clone(), k);

    println!("Arnoldi Iteration using Classical GS:");
    // println!("V:\n{:?}", V);
    // println!("H:\n{:?}", H);

    let is_orthonormal = are_columns_orthonormal(&V);

    if is_orthonormal {
        println!("The columns are orthonormal.");
    } else {
        println!("The columns are not orthonormal.");
    }

    // Arnoldi iteration Modified GS
    let (V, H) = arnoldi_mg_iteration(&A, r0.clone(), k);

    println!("Arnoldi Iteration using Modified GS:");
    // println!("V:\n{:?}", V);
    // println!("H:\n{:?}", H);

    let is_orthonormal = are_columns_orthonormal(&V);

    if is_orthonormal {
        println!("The columns are orthonormal.");
    } else {
        println!("The columns are not orthonormal.");
    }
    
}