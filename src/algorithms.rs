use nalgebra::{DMatrix, DVector, Vector};



pub fn gs(A: &DMatrix<f64>, mut v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>){
    /**
    Performs the Gram-Schmidt orthogonalization process.

    Args:
    - A (&DMatrix<f64>): Input matrix.
    - v (DVector<f64>): Initial vector.
    - k (usize): Number of iterations.

    Returns:
    - (DMatrix<f64>, DMatrix<f64>): Tuple containing the orthogonalized matrix V and the triangular matrix H.
    */

    let n = A.ncols();
    
    let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(n, k+1);
    V.column_mut(0).copy_from(&v.unscale(v.norm()));
 
    let mut H: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(k+1, k); 
    for j in 1..k+1 {
        v = A*v;
        let mut vtilde = v.clone();
        for i in 0..j {
            H[(i,j-1)] = V.column(i).conjugate().dot(&v);
            vtilde = vtilde - H[(i,j-1)]*&V.column(i);
        }
            
        H[(j-1,j-1)] = vtilde.norm();
        V.set_column(j, &(vtilde/H[(j-1,j-1)]));
    }
    return (V, H);
}

pub fn arnoldi_cgs(A: &DMatrix<f64>, mut v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>){
    /**
    Performs the Arnoldi orthogonalization process using Classical Gram-Schmidt.

    Args:
    - A (&DMatrix<f64>): Input matrix.
    - v (DVector<f64>): Initial vector.
    - k (usize): Number of iterations.

    Returns:
    - (DMatrix<f64>, DMatrix<f64>): Tuple containing the orthogonalized matrix V and the upper Hessenberg matrix H.
    */

    let n = A.ncols();
    
    let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(n, k+1);
    let mut H: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(k+1, k); 
    
    V.column_mut(0).copy_from(&v.unscale(v.norm()));

    for j in 1..k+1 {
        let av = A*V.column(j-1);
        let mut vtilde = av.clone();
        for i in 0..j {
            H[(i,j-1)] = V.column(i).conjugate().dot(&av);
            vtilde = vtilde - H[(i,j-1)]*&V.column(i);
        }
            
        H[(j,j-1)] = vtilde.norm();
        V.set_column(j, &(vtilde/H[(j,j-1)]));
    }
    return (V, H);
}

pub fn arnoldi_mgs(A: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    /**
    Performs the Arnoldi orthogonalization process using Modified Gram-Schmidt.

    Args:
    - A (&DMatrix<f64>): Input matrix.
    - v (DVector<f64>): Initial vector.
    - k (usize): Number of iterations.

    Returns:
    - (DMatrix<f64>, DMatrix<f64>): Tuple containing the orthogonalized matrix V and the upper Hessenberg matrix H.
    */

    let n = A.ncols();

    let mut V = DMatrix::zeros(n, k+1);
    let mut H = DMatrix::zeros(k+1, k);
    
    V.column_mut(0).copy_from(&v.unscale(v.norm()));

    for j in 1..k+1 {
        V.set_column(j, &(A*V.column(j-1)));
        for i in 0..j {
            H[(i, j - 1)] = V.column(i).conjugate().dot(&V.column(j));
            V.set_column(j, &(V.column(j)-H[(i, j-1)]*V.column(i)));
        }  // Subtract the projections on previous vectors
        for i in 0..j {
            H[(i, j - 1)] = H[(i, j-1)] + V.column(i).conjugate().dot(&V.column(j));
            V.set_column(j, &(V.column(j)-(V.column(i).conjugate().dot(&V.column(j)))*V.column(i)));
        }  // Subtract the projections on previous vectors
        H[(j, j-1)] = V.column(j).norm();
        V.set_column(j, &(V.column(j) / H[(j, j-1)])); //Right now the only difference is here
    }
            
    return (V, H);
}

pub fn lanczos(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    /**
    Performs the Lanczos orthogonalization process.

    Args:
    - A (&DMatrix<f64>): Input matrix.
    - r0 (DVector<f64>): Initial vector.
    - n (usize): Number of iterations.

    Returns:
    - (DMatrix<f64>, DMatrix<f64>): Tuple containing the orthogonalized matrix V and the upper Hessenberg matrix H.
    */
    
    let mut H = DMatrix::zeros(n+1, n);
    let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(A.nrows(), n+1);


    V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

    for k in 1..n+1 {
        let mut u = A * &V.column(k - 1);  // Generate a new candidate vector
        if k!=1{
            u = A * &V.column(k - 1) - &V.column(k - 2).scale(H[(k-1, k-2)])
        }
        H[(k-1, k-1)] = *(&V.column(k - 1).transpose()*&u).get((0, 0)).unwrap(); // Diagonal elements
        let mut vhat = u-&V.column(k - 1).scale(H[(k-1, k-1)]);

        H[(k, k-1)] = vhat.norm(); // Subdiagonal elements 1,0 2,1 3,2...

        if k!=n{
            H[(k-1, k)] = H[(k, k-1)]
        }
        V.set_column(k, &(vhat/H[(k, k-1)]));
    }
            
    return (V, H);
}