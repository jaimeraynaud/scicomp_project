#![allow(warnings)]


use nalgebra::{DMatrix, DVector, Vector};
use std::io::prelude::*;
use std::path::Path;
use std::fs::File;
use csv::WriterBuilder;
use std::error::Error;
use std::time::{Duration, Instant};

fn gs_iteration(A: &DMatrix<f64>, mut v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>){
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

// fn arnoldi_cg_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
//     let mut H = DMatrix::zeros(n+1, n);
//     let mut V: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Dyn, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Dyn>> = DMatrix::zeros(A.nrows(), n+1);

//     V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

//     for k in 1..n+1 {
//         let mut v_ = A * &V.column(k - 1);  // Generate a new candidate vector
//         for i in 0..k {
//             H[(i, k - 1)] = (A * &V.column(k - 1)).conjugate().dot(&V.column(i));
//             v_ = v_ - H[(i, k - 1)]*&V.column(i);
//         }  // Subtract the projections on previous vectors
//         H[(k, k-1)] = v_.norm();
//         V.set_column(k, &(v_ / H[(k, k-1)]));
//     }
            
//     return (V, H);
// }

fn arnoldi_cg_iteration(A: &DMatrix<f64>, mut v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>){
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
            
        H[(j-1,j-1)] = vtilde.norm();
        V.set_column(j, &(vtilde/H[(j-1,j-1)]));
    }
    return (V, H);
}

fn arnoldi_mg_iteration(A: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
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

// fn arnoldi_mg_iteration(A: &DMatrix<f64>, v: DVector<f64>, k: usize) -> (DMatrix<f64>, DMatrix<f64>) {
//     let n = A.ncols();

//     let mut V = DMatrix::zeros(n, k+1);
//     let mut H = DMatrix::zeros(k+1, k);
    
//     V.column_mut(0).copy_from(&v.unscale(v.norm()));

//     for j in 1..k+1 {
//         let mut v_: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>> = A * &V.column(j - 1);  // Generate a new candidate vector
//         for i in 0..j {
//             H[(i, j - 1)] = V.column(j).conjugate().dot(&v_);
//             v_ = v_ - H[(i, j - 1)] * V.column(i);
//         }  // Subtract the projections on previous vectors
//         H[(j, j-1)] = v_.norm();
//         V.set_column(k, &(v_ / H[(j, j-1)])); //Right now the only difference is here
//     }
            
//     return (V, H);
// }


fn lanczos_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
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

fn write_matrices_to_csv(V_gs: &DMatrix<f64>, R_gs: &DMatrix<f64>, V_cgs: &DMatrix<f64>, H_cgs: &DMatrix<f64>, V_mgs: &DMatrix<f64>, H_mgs: &DMatrix<f64>, V_lanczos: &DMatrix<f64>, T_lanczos: &DMatrix<f64>, k: usize) -> Result<(), std::io::Error> {
    
    write_matrix_to_csv(V_gs, &format!("./experiment_results/gs/V/V_gs_{}.csv", k))?;
    write_matrix_to_csv(R_gs, &format!("./experiment_results/gs/R/R_gs_{}.csv", k))?;
    write_matrix_to_csv(V_cgs, &format!("./experiment_results/cgs/V/V_cgs_{}.csv", k))?;
    write_matrix_to_csv(H_cgs, &format!("./experiment_results/cgs/H/H_cgs_{}.csv", k))?;
    write_matrix_to_csv(V_mgs, &format!("./experiment_results/mgs/V/V_mgs_{}.csv", k))?;
    write_matrix_to_csv(H_mgs, &format!("./experiment_results/mgs/H/H_mgs_{}.csv", k))?;
    write_matrix_to_csv(V_lanczos, &format!("./experiment_results/lanczos/V/V_lanczos_{}.csv", k))?;
    write_matrix_to_csv(T_lanczos, &format!("./experiment_results/lanczos/T/T_lanczos_{}.csv", k))?;
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

fn write_orthogonality_vectors_to_csv(orthogonality_loss_gs_vec: &Vec<f64>, orthogonality_loss_cgs_vec: &Vec<f64>, orthogonality_loss_mgs_vec: &Vec<f64>, orthogonality_loss_mgs_H_vec: &Vec<f64>, orthogonality_loss_lanczos_vec: &Vec<f64>) -> Result<(), std::io::Error> {
    write_vector_to_csv(orthogonality_loss_gs_vec, "./experiment_results/gs/orthogonality_loss_gs_vec.csv");
    write_vector_to_csv(orthogonality_loss_cgs_vec, "./experiment_results/cgs/orthogonality_loss_cgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_vec, "./experiment_results/mgs/orthogonality_loss_mgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_H_vec, "./experiment_results/mgs/orthogonality_loss_mgs_H_vec.csv");
    write_vector_to_csv(orthogonality_loss_lanczos_vec, "./experiment_results/lanczos/orthogonality_loss_lanczos_vec.csv");
    Ok(())
}


fn measure_avg_execution_time<F>(f: F) -> Duration
where
    F: Fn() -> (),
{
    let num_iterations = 2; // Change here to modify the number of iterations to average the results.
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

fn compute_time(A: &DMatrix<f64>, A_H: &DMatrix<f64>, r0: DVector<f64>, k_max: usize){
    let mut time_gs_vec: Vec<f64> = Vec::new();
    let mut time_cgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_H_vec: Vec<f64> = Vec::new();
    let mut time_lanczos_vec: Vec<f64> = Vec::new();
    

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    for k in &k_vector{
        println!("Iteration {:?}/{:?}", k, k_max);
        let averaged_time_gs = measure_avg_execution_time(|| {gs_iteration(&A, r0.clone(), *k);});
        time_gs_vec.push(averaged_time_gs.as_secs_f64());

        let averaged_time_cgs = measure_avg_execution_time(|| {arnoldi_cg_iteration(&A, r0.clone(), *k);});
        time_cgs_vec.push(averaged_time_cgs.as_secs_f64());

        let averaged_time_mgs = measure_avg_execution_time(|| {arnoldi_mg_iteration(&A, r0.clone(), *k);});
        time_mgs_vec.push(averaged_time_mgs.as_secs_f64());

        let averaged_time_mgs_H = measure_avg_execution_time(|| {arnoldi_mg_iteration(&A_H, r0.clone(), *k);});
        time_mgs_H_vec.push(averaged_time_mgs_H.as_secs_f64());

        let averaged_time_lanczos = measure_avg_execution_time(|| {lanczos_iteration(&A_H, r0.clone(), *k);});
        time_lanczos_vec.push(averaged_time_lanczos.as_secs_f64());
    }

    write_vector_to_csv(&time_gs_vec, "./experiment_results/gs/time_gs_vec.csv");
    write_vector_to_csv(&time_cgs_vec, "./experiment_results/cgs/time_cgs_vec.csv");
    write_vector_to_csv(&time_mgs_vec, "./experiment_results/mgs/time_mgs_vec.csv");
    write_vector_to_csv(&time_mgs_H_vec, "./experiment_results/mgs/time_mgs_H_vec.csv");
    write_vector_to_csv(&time_lanczos_vec, "./experiment_results/lanczos/time_lanczos_vec.csv");

}


fn orchestrator() {
    println!("\n===========================================================================================================================");
    println!("Comparison of algorithms for computing orthonormal bases in Rust");
    println!("===========================================================================================================================\n");

    // Set the number of Arnoldi iterations
    let n = 75; // Size of the initial matrix A (n>k_max)
    let k_max = 70; // 200 (n > k_max)


    println!("===========================================================================================================================");
    println!("Initial values: n (size of the matrix) = {:?}, k (dimension of the subspace) = {:?}", n, k_max);
    println!("===========================================================================================================================\n");

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    let A_H = A.clone()+A.transpose();

    let _ = write_matrix_to_csv(&A, &format!("./experiment_results/A.csv"));
    let _ = write_matrix_to_csv(&A_H, &format!("./experiment_results/A_H.csv"));

    // Choose a random initial vector and normalize it
    let mut r0 = DVector::new_random(n);
    r0.normalize_mut();

    let vec_r0: Vec<f64> = r0.iter().cloned().collect();
    let _ = write_vector_to_csv(&vec_r0, "./experiment_results/r0.csv");

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
        let (V_gs, R_gs) = gs_iteration(&A, r0.clone(), *k);
        orthogonality_loss_gs_vec.push(orthogonality_loss(&V_gs));

        // Arnoldi iteration Classical GS
        let (V_cgs, H_cgs) = arnoldi_cg_iteration(&A, r0.clone(), *k);
        orthogonality_loss_cgs_vec.push(orthogonality_loss(&V_cgs));

        // Arnoldi iteration Modified GS
        let (V_mgs, H_mgs) = arnoldi_mg_iteration(&A, r0.clone(), *k);
        orthogonality_loss_mgs_vec.push(orthogonality_loss(&V_mgs));

        // Arnoldi iteration Modified GS (Hermitian matrix)
        let (V_mgs_H, H_mgs_H) = arnoldi_mg_iteration(&A_H, r0.clone(), *k);
        orthogonality_loss_mgs_H_vec.push(orthogonality_loss(&V_mgs_H));

        // Arnoldi iteration Modified GS
        let (V_lanczos, T_lanczos) = lanczos_iteration(&A_H, r0.clone(), *k);
        orthogonality_loss_lanczos_vec.push(orthogonality_loss(&V_lanczos));
        
        // Write orthogonality loss and time vectors to csv
        write_orthogonality_vectors_to_csv(&orthogonality_loss_gs_vec, &orthogonality_loss_cgs_vec, &orthogonality_loss_mgs_vec, &orthogonality_loss_mgs_H_vec, &orthogonality_loss_lanczos_vec);

        // Write matrices to CSV files
        write_matrices_to_csv(&V_gs, &R_gs, &V_cgs, &H_cgs, &V_mgs, &H_mgs, &V_lanczos, &T_lanczos, *k);

        //println!("Are Lanczos columns for {} orthonormal? {:?}", k, are_columns_orthonormal(&V_lanczos));

    } // End of k loop
    println!("===========================================================================================================================\n");

    // Compute and write averaged execution time: CAUTION! Long execution time
    println!("===========================================================================================================================");
    println!("Computing average runtime...");
    compute_time(&A, &A_H,r0, k_max);
    println!("===========================================================================================================================\n");

    println!("Finished\n")


}



fn main() {
    orchestrator()
}