use nalgebra::{DMatrix, DVector, MatrixMN, U1, Dynamic};
use nalgebra::linalg::QR;

fn arnoldi_iteration(matrix: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut h = DMatrix::zeros(k + 1, k);
    let mut v_collection = DMatrix::zeros(matrix.nrows(), k + 1);

    v_collection.column_mut(0).copy_from(&v);

    for j in 0..k {
        let mut q = matrix * v_collection.column(j);
        for i in 0..=j {
            h[(i, j)] = q.dot(&v_collection.column(i));
            q -= h[(i, j)] * &v_collection.column(i);
        }
        h[(j + 1, j)] = q.norm();
        if h[(j + 1, j)] != 0.0 {
            v_collection.column_mut(j + 1).copy_from(&(q / h[(j + 1, j)]).into_owned());

        }
    }

    (v_collection, h)
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
        let norm = DVector::norm(&v);
        if norm.abs() > 1e-10 {
            q.set_column(j, &(v / norm));
        } else {
            q.set_column(j, &v);
        }
    }

    q
}

fn are_columns_orthonormal(matrix: &DMatrix<f64>) -> bool{
    let num_columns = matrix.ncols();

    for i in 0..num_columns {
        for j in 0..i {
            let col_i = matrix.column(i);
            println!("Maginitude = {}",col_i.magnitude());
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

fn arnoldi_cgs(matrix: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut h = DMatrix::zeros(k + 1, k);
    let mut v_collection = DMatrix::zeros(matrix.nrows(), k + 1);

    v_collection.column_mut(0).copy_from(&v);

    for j in 0..k {
        let mut q = matrix * v_collection.column(j);
        for i in 0..=j {
            h[(i, j)] = q.dot(&v_collection.column(i));
            q -= h[(i, j)] * &v_collection.column(i);
        }
        h[(j + 1, j)] = q.norm();
        if h[(j + 1, j)] != 0.0 {
            v_collection.column_mut(j + 1).copy_from(&(q / h[(j + 1, j)]).into_owned());
        }
    }

    (v_collection, h)
}

fn arnoldi_mgs(matrix: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut h = DMatrix::zeros(k + 1, k);
    let mut v_collection = DMatrix::zeros(matrix.nrows(), k + 1);

    v_collection.column_mut(0).copy_from(&v);

    for j in 0..k {
        let mut v = matrix.column(j).clone_owned();
        for i in 0..j {
            let r = v_collection.column(i).dot(&v);
            v -= r * &v_collection.column(i);
            h[(i, j)] = r;
        }
        let norm = v.norm();
        if norm.abs() > 1e-10 {
            v_collection.column_mut(j + 1).copy_from(&(v / norm));
        } else {
            v_collection.column_mut(j + 1).copy_from(&v);
        }
    }

    (v_collection, h)
}

fn main() {
    // Example usage:
    let matrix = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let k = 2; // Number of iterations or desired dimension of Krylov subspace

    // Choose a random initial vector for Arnoldi algorithm
    let v0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);

    // Arnoldi iteration with Classical Gram-Schmidt
    let (v_collection_cgs, h_cgs) = arnoldi_cgs(&matrix, v0.clone(), k);

    println!("Arnoldi Iteration with Classical Gram-Schmidt:");
    println!("V:\n{:?}", v_collection_cgs);
    println!("H:\n{:?}", h_cgs);

    // Arnoldi iteration with Modified Gram-Schmidt
    let (v_collection_mgs, h_mgs) = arnoldi_mgs(&matrix, v0.clone(), k);

    println!("\nArnoldi Iteration with Modified Gram-Schmidt:");
    println!("V:\n{:?}", v_collection_mgs);
    println!("H:\n{:?}", h_mgs);
}

// fn main() {
//     // Define your matrix here
//     let a = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

//     // Set the number of Arnoldi iterations
//     let k = 10;

//     // Choose a random initial vector for Arnoldi algorithm
//     let v0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);

//     // Arnoldi iteration
//     let (v_collection, h) = arnoldi_iteration(&a, v0, k);

//     println!("Arnoldi Iteration:");
//     println!("V:\n{:?}", v_collection);
//     println!("H:\n{:?}", h);

//     // Classical Gram-Schmidt
//     let q_classical = classical_gram_schmidt(&a);

//     println!("\nClassical Gram-Schmidt:");
//     println!("Q:\n{:?}", q_classical);

//     let is_orthonormal_class = are_columns_orthonormal(&q_classical);

//     if is_orthonormal_class {
//         println!("The columns are orthonormal.");
//     } else {
//         println!("The columns are not orthonormal.");
//     }

//     // Modified Gram-Schmidt
//     let q_modified = modified_gram_schmidt(&a);

//     println!("\nModified Gram-Schmidt:");
//     println!("Q:\n{:?}", q_modified);

//     let is_orthonormal_mod = are_columns_orthonormal(&q_modified);

//     if is_orthonormal_mod {
//         println!("The columns are orthonormal.");
//     } else {
//         println!("The columns are not orthonormal.");
//     }
    
// }
