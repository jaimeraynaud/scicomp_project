use nalgebra::{DMatrix, DVector, Vector, U1};
use std::io::prelude::*;
use std::path::Path;
use std::fs::File;
use csv::WriterBuilder;
use std::error::Error;
use std::time::{Duration, Instant};

fn gs_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
    let mut R = DMatrix::zeros(n+1, n);
    let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(A.nrows(), n+1);

    //R[(0, 0)] = &r0.unscale(r0.norm());
    V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

    for k in 1..n+1 {
        let mut v_ = A.pow((k-1).try_into().unwrap()) * &V.column(0);  // Generate a new candidate vector
        for i in 0..k {
            R[(i, k)] = (A.pow((k-1).try_into().unwrap()) * &V.column(0)).conjugate().dot(&V.column(i));
            v_ = v_ - R[(i, k)]*&V.column(i);
        }  // Subtract the projections on previous vectors
        R[(k, k)] = v_.norm();

        if R[(k, k)] > eps{
            V.set_column(k, &(v_ / R[(k, k)]));
        } else { // Add the produced vector to the list, unless 
            return (V, R);
        }  // If that happens, stop iterating.
    }
            
    return (V, R);
}

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

fn write_matrix_to_csv(matrix: &DMatrix<f64>, filename: &str) -> Result<(), std::io::Error> {
    let file_path = Path::new(filename);
    let mut file = File::create(file_path)?;

    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let value = matrix[(i, j)];
            write!(file, "{},", value)?;
        }
        writeln!(file, "")?;
    }

    Ok(())
}

fn write_matrices_to_csv(V_gs: &DMatrix<f64>, R_gs: &DMatrix<f64>, V_cgs: &DMatrix<f64>, H_cgs: &DMatrix<f64>, V_mgs: &DMatrix<f64>, H_mgs: &DMatrix<f64>, k: usize) -> Result<(), std::io::Error> {
    
    write_matrix_to_csv(V_gs, &format!("../python_project/gs/V/V_gs_{}.csv", k))?;
    write_matrix_to_csv(R_gs, &format!("../python_project/gs/R/R_gs_{}.csv", k))?;
    write_matrix_to_csv(V_cgs, &format!("../python_project/cgs/V/V_cgs_{}.csv", k))?;
    write_matrix_to_csv(H_cgs, &format!("../python_project/cgs/H/H_cgs_{}.csv", k))?;
    write_matrix_to_csv(V_mgs, &format!("../python_project/mgs/V/V_mgs_{}.csv", k))?;
    write_matrix_to_csv(H_mgs, &format!("../python_project/mgs/H/H_mgs_{}.csv", k))?;

    Ok(())
}

fn write_vector_to_csv(data: &Vec<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(file_path)?;
    let mut writer = WriterBuilder::new().from_writer(file);

    for value in data {
        writer.write_record(&[value.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}

fn write_orthogonality_vectors_to_csv(orthogonality_loss_gs_vec: &Vec<f64>, orthogonality_loss_cgs_vec: &Vec<f64>, orthogonality_loss_mgs_vec: &Vec<f64>) -> Result<(), std::io::Error> {
    write_vector_to_csv(orthogonality_loss_gs_vec, "../python_project/gs/orthogonality_loss_gs_vec.csv");
    write_vector_to_csv(orthogonality_loss_cgs_vec, "../python_project/cgs/orthogonality_loss_cgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_vec, "../python_project/mgs/orthogonality_loss_mgs_vec.csv");
    Ok(())
}


fn measure_avg_execution_time<F>(f: F) -> Duration
where
    F: Fn() -> (),
{
    let num_iterations = 100; // Change here to modify the number of iterations to average the results.
    let mut total_duration = Duration::from_secs(0);

    for _ in 0..num_iterations {
        let start_time = Instant::now();
        f();
        let end_time = Instant::now();
        total_duration += end_time - start_time;
    }
    let averaged_duration = total_duration / num_iterations as u32;
    println!("{:?}", averaged_duration);
    return averaged_duration
}

fn compute_time(A: &DMatrix<f64>, r0: DVector<f64>, k_max: usize){
    let mut time_gs_vec: Vec<f64> = Vec::new();
    let mut time_cgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_vec: Vec<f64> = Vec::new();

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    for k in &k_vector{
        let averaged_time_gs = measure_avg_execution_time(|| {gs_iteration(&A, r0.clone(), *k);});
        time_gs_vec.push(averaged_time_gs.as_secs_f64());

        let averaged_time_cgs = measure_avg_execution_time(|| {arnoldi_cg_iteration(&A, r0.clone(), *k);});
        time_cgs_vec.push(averaged_time_cgs.as_secs_f64());

        let averaged_time_mgs = measure_avg_execution_time(|| {arnoldi_mg_iteration(&A, r0.clone(), *k);});
        time_mgs_vec.push(averaged_time_mgs.as_secs_f64());
    }

    write_vector_to_csv(&time_gs_vec, "../python_project/gs/time_gs_vec.csv");
    write_vector_to_csv(&time_cgs_vec, "../python_project/cgs/time_cgs_vec.csv");
    write_vector_to_csv(&time_mgs_vec, "../python_project/mgs/time_mgs_vec.csv");

}

fn orchestrator() {
    // Set the number of Arnoldi iterations
    let n = 50; // The number of iterations give the number of orthonormal columns

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    // Choose a random initial vector for Arnoldi algorithm
    let r0 = DVector::new_random(n);

    let k_max = 45;

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    let mut orthogonality_loss_gs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_cgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_vec: Vec<f64> = Vec::new();

    for k in &k_vector{

        // GS iteration
        let (V_gs, R_gs) = gs_iteration(&A, r0.clone(), *k);
        orthogonality_loss_gs_vec.push(orthogonality_loss(&V_gs));

        // Arnoldi iteration Classical GS
        let (V_cgs, H_cgs) = arnoldi_cg_iteration(&A, r0.clone(), *k);
        orthogonality_loss_cgs_vec.push(orthogonality_loss(&V_cgs));

        // Arnoldi iteration Modified GS
        let (V_mgs, H_mgs) = arnoldi_mg_iteration(&A, r0.clone(), *k);
        orthogonality_loss_mgs_vec.push(orthogonality_loss(&V_mgs));
        
        // Write orthogonality loss and time vectors to csv
        write_orthogonality_vectors_to_csv(&orthogonality_loss_gs_vec, &orthogonality_loss_cgs_vec, &orthogonality_loss_mgs_vec);

        // Write matrices to CSV files
        write_matrices_to_csv(&V_gs, &R_gs, &V_cgs, &H_cgs, &V_mgs, &H_mgs, *k);

        println!("Are GS columns for {} orthonormal? {:?}", k, are_columns_orthonormal(&V_gs));

    } // End of k loop

    println!("\nk: {}", k_max);
    println!("\nn: {}", n);
    println!("\nArnoldi Iteration using Classical GS orthogonality loss:");
    println!("{:?}", orthogonality_loss_cgs_vec.clone());

    println!("\nArnoldi Iteration using Modified GS orthogonality loss:");
    println!("{:?}", orthogonality_loss_mgs_vec);

    // Compute and write averaged execution time: CAUTION! Long execution time
    // compute_time(&A, r0, k_max);

}



fn main() {
    orchestrator()
}