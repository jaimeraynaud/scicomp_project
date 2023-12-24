use nalgebra::{DMatrix, DVector};

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
            let factor = matrix.column(i).dot(&v) / matrix.column(i).dot(&matrix.column(i));
            v -= factor * &matrix.column(i);
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
            let r = q.column(i).dot(&v) / q.column(i).dot(&q.column(i));
            v -= r * &q.column(i);
        }
        q.set_column(j, &v.normalize());
    }

    q
}

fn main() {
    // Define your matrix here
    let a = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    // Set the number of Arnoldi iterations
    let k = 2;

    // Choose a random initial vector for Arnoldi algorithm
    let v0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);

    // Arnoldi iteration
    let (v_collection, h) = arnoldi_iteration(&a, v0, k);

    println!("Arnoldi Iteration:");
    println!("V:\n{:?}", v_collection);
    println!("H:\n{:?}", h);

    // Classical Gram-Schmidt
    let q_classical = classical_gram_schmidt(&a);

    println!("\nClassical Gram-Schmidt:");
    println!("Q:\n{:?}", q_classical);

    // Modified Gram-Schmidt
    let q_modified = modified_gram_schmidt(&a);

    println!("\nModified Gram-Schmidt:");
    println!("Q:\n{:?}", q_modified);
}
