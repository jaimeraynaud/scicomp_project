use std::ops::Div;

use nalgebra::{DMatrix, DVector, Vector, MatrixMN, U1, Dynamic, Vector1};
use nalgebra::linalg::QR;
use ndarray::{Array2, ArrayView1};


fn arnoldi_iteration(A: &DMatrix<f64>, b: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
    let mut h = DMatrix::zeros(n+1, n);
    let mut Q = DMatrix::zeros(A.nrows(), n+1);


    Q.column_mut(0).copy_from(&b.unscale(b.norm()));

    for k in 1..n+1 {
        let mut v = A * &Q.column(k - 1);  // Generate a new candidate vector
        for j in 0..k {
            h[(j, k - 1)] = Q.column(j).conjugate().dot(&v);
            v = v - h[(j, k - 1)] * Q.column(j);
        }  // Subtract the projections on previous vectors
        h[(k, k-1)] = v.norm();

        if h[(k, k-1)] > eps{
            Q.set_column(k, &(v / h[(k, k-1)]));
        } else { // Add the produced vector to the list, unless 
            return (Q, h);
        }  // If that happens, stop iterating.
    }
            
    return (Q, h);
}


fn classical_gram_schmidt(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut q = DMatrix::zeros(matrix.nrows(), matrix.ncols());

    for j in 0..matrix.ncols() {
        let mut v = matrix.column(j).clone_owned();
        for i in 0..j {
            let r = q.column(i).dot(&v) / q.column(i).dot(&q.column(i));
            v -= r * &q.column(i);
        }
        q.set_column(j, &v.normalize());
    }

    q
}

fn modified_gram_schmidt(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let mut q = DMatrix::zeros(matrix.nrows(), matrix.ncols());

    for j in 0..matrix.ncols() {
        let mut v = matrix.column(j).clone_owned();
        for i in 0..j {
            let r = q.column(i).dot(&v);
            v -= r * &q.column(i);
        }

        // Use nalgebra's QR decomposition to normalize the vector
        let norm = Vector::norm(&v);
        if norm.abs() > 1e-10 {
            q.set_column(j, &(v / norm));
        } else {
            q.set_column(j, &v);
        }
    }

    q
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
    // Define your matrix here
    let A = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    // Set the number of Arnoldi iterations
    let n = 2; // Seems like the number of iterations give the number of orthonormal columns

    // Choose a random initial vector for Arnoldi algorithm
    let b = DVector::from_vec(vec![1.0, 1.0, 1.0]);

    // Arnoldi iteration
    let (v_collection, h) = arnoldi_iteration(&A, b, n);

    println!("Arnoldi Iteration:");
    println!("V:\n{:?}", v_collection);
    println!("H:\n{:?}", h);

    let is_orthonormal = are_columns_orthonormal(&v_collection);

    if is_orthonormal {
        println!("The columns are orthonormal.");
    } else {
        println!("The columns are not orthonormal.");
    }

    // // Classical Gram-Schmidt
    // let q_classical = classical_gram_schmidt(&a);

    // println!("\nClassical Gram-Schmidt:");
    // println!("Q:\n{:?}", q_classical);

    // let is_orthonormal_class = are_columns_orthonormal(&q_classical);

    // if is_orthonormal_class {
    //     println!("The columns are orthonormal.");
    // } else {
    //     println!("The columns are not orthonormal.");
    // }

    // // Modified Gram-Schmidt
    // let q_modified = modified_gram_schmidt(&a);

    // println!("\nModified Gram-Schmidt:");
    // println!("Q:\n{:?}", q_modified);

    // let is_orthonormal_mod = are_columns_orthonormal(&q_modified);

    // if is_orthonormal_mod {
    //     println!("The columns are orthonormal.");
    // } else {
    //     println!("The columns are not orthonormal.");
    // }
    
}