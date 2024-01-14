use nalgebra::{DMatrix, DVector, Vector, U1};
use plotters::prelude::*;
use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};



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

fn orthogonality_loss(V: &DMatrix<f64>) -> f64 {
    let T = V.transpose()*V;
    let I = DMatrix::<f64>::identity(T.nrows(), T.ncols());
    return (I-T).norm()
}

fn visualize(k: Vec<usize>, values1: Vec<f64>, values2: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate the y-axis range based on the minimum and maximum values of values1 and values2
    let min_y = values1.iter().copied().chain(values2.iter().copied()).fold(f64::INFINITY, f64::min);
    let max_y = values1.iter().copied().chain(values2.iter().copied()).fold(f64::NEG_INFINITY, f64::max);

    // Create a new figure
    let mut fg = Figure::new();

    // Plot values1
    fg.axes2d()
        .lines(k.iter(), values1.iter(), &[Caption("Values 1"), Color("red")])
        .set_y_range(Fix(min_y), Fix(max_y));

    // Plot values2
    fg.axes2d()
        .lines(k.iter(), values2.iter(), &[Caption("Values 2"), Color("blue")])
        .set_y_range(Fix(min_y), Fix(max_y));

    // Set x-axis label
    fg.axes2d().set_x_label("k", &[]);

    // Display the plot
    fg.show()?;
    Ok(())
}

fn main() {
    // Set the number of Arnoldi iterations
    let n = 50; // The number of iterations give the number of orthonormal columns

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    // Choose a random initial vector for Arnoldi algorithm
    let r0 = DVector::new_random(n);

    let k_vector: Vec<usize> = (2..=10).collect(); // k = Grade of the vector b

    let mut orthogonality_loss_cgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_vec: Vec<f64> = Vec::new();

    for k in &k_vector{

        // Arnoldi iteration Classical GS
        let (V_cgs, H_cgs) = arnoldi_cg_iteration(&A, r0.clone(), *k);

        // println!("\nArnoldi Iteration using Classical GS:");

        // let is_orthonormal = are_columns_orthonormal(&V);

        // if is_orthonormal {
        //     println!("The columns are orthonormal.");
        // } else {
        //     println!("The columns are not orthonormal.");
        // }
        
        orthogonality_loss_cgs_vec.push(orthogonality_loss(&V_cgs));

        // Arnoldi iteration Modified GS
        let (V_mgs, H_mgs) = arnoldi_mg_iteration(&A, r0.clone(), *k);

        // println!("\nArnoldi Iteration using Modified GS:");

        // let is_orthonormal = are_columns_orthonormal(&V);

        // if is_orthonormal {
        //     println!("The columns are orthonormal.");
        // } else {
        //     println!("The columns are not orthonormal.");
        // }

        orthogonality_loss_mgs_vec.push(orthogonality_loss(&V_mgs));

    } // End of k loop
    println!("\nArnoldi Iteration using Classical GS:");
    println!("{:?}", orthogonality_loss_cgs_vec);

    println!("\nArnoldi Iteration using Modified GS:");
    println!("{:?}", orthogonality_loss_mgs_vec);
    
    // Call the visualize function
    if let Err(err) = visualize(k_vector, orthogonality_loss_cgs_vec, orthogonality_loss_mgs_vec) {
        eprintln!("Error: {}", err);
    }
}