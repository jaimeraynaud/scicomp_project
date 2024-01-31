use nalgebra::{DMatrix, DVector, Vector, U1};
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
        V.set_column(k, &(v_ / H[(k, k-1)]));
    }
            
    return (V, H);
}

fn arnoldi_mg_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
    let mut H = DMatrix::zeros(n+1, n);
    let mut V = DMatrix::zeros(A.nrows(), n+1);


    V.column_mut(0).copy_from(&r0.unscale(r0.norm()));

    for k in 1..n+1 {
        let mut v_: nalgebra::Matrix<f64, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>> = A * &V.column(k - 1);  // Generate a new candidate vector
        for j in 0..k {
            H[(j, k - 1)] = V.column(j).conjugate().dot(&v_);
            v_ = v_ - H[(j, k - 1)] * V.column(j);
        }  // Subtract the projections on previous vectors
        H[(k, k-1)] = v_.norm();
        V.set_column(k, &(v_ / H[(k, k-1)]));
    }
            
    return (V, H);
}


fn lanczos_iteration(A: &DMatrix<f64>, r0: DVector<f64>, n: usize) -> (DMatrix<f64>, DMatrix<f64>) {
    let eps = 1e-12;
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
    
    write_matrix_to_csv(V_gs, &format!("./python_visualization/gs/V/V_gs_{}.csv", k))?;
    write_matrix_to_csv(R_gs, &format!("./python_visualization/gs/R/R_gs_{}.csv", k))?;
    write_matrix_to_csv(V_cgs, &format!("./python_visualization/cgs/V/V_cgs_{}.csv", k))?;
    write_matrix_to_csv(H_cgs, &format!("./python_visualization/cgs/H/H_cgs_{}.csv", k))?;
    write_matrix_to_csv(V_mgs, &format!("./python_visualization/mgs/V/V_mgs_{}.csv", k))?;
    write_matrix_to_csv(H_mgs, &format!("./python_visualization/mgs/H/H_mgs_{}.csv", k))?;
    write_matrix_to_csv(V_lanczos, &format!("./python_visualization/lanczos/V/V_lanczos_{}.csv", k))?;
    write_matrix_to_csv(T_lanczos, &format!("./python_visualization/lanczos/T/T_lanczos_{}.csv", k))?;
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
    write_vector_to_csv(orthogonality_loss_gs_vec, "./python_visualization/gs/orthogonality_loss_gs_vec.csv");
    write_vector_to_csv(orthogonality_loss_cgs_vec, "./python_visualization/cgs/orthogonality_loss_cgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_vec, "./python_visualization/mgs/orthogonality_loss_mgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_H_vec, "./python_visualization/mgs/orthogonality_loss_mgs_H_vec.csv");
    write_vector_to_csv(orthogonality_loss_lanczos_vec, "./python_visualization/lanczos/orthogonality_loss_lanczos_vec.csv");
    Ok(())
}


fn measure_avg_execution_time<F>(f: F) -> Duration
where
    F: Fn() -> (),
{
    let num_iterations = 10; // Change here to modify the number of iterations to average the results.
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

fn compute_time(A: &DMatrix<f64>, A_H: &DMatrix<f64>, r0: DVector<f64>, k_max: usize){
    let mut time_gs_vec: Vec<f64> = Vec::new();
    let mut time_cgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_vec: Vec<f64> = Vec::new();
    let mut time_mgs_H_vec: Vec<f64> = Vec::new();
    let mut time_lanczos_vec: Vec<f64> = Vec::new();
    

    let k_vector: Vec<usize> = (2..=k_max).collect(); // k = Grade of the vector b

    for k in &k_vector{
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

    write_vector_to_csv(&time_gs_vec, "./python_visualization/gs/time_gs_vec.csv");
    write_vector_to_csv(&time_cgs_vec, "./python_visualization/cgs/time_cgs_vec.csv");
    write_vector_to_csv(&time_mgs_vec, "./python_visualization/mgs/time_mgs_vec.csv");
    write_vector_to_csv(&time_mgs_H_vec, "./python_visualization/mgs/time_mgs_H_vec.csv");
    write_vector_to_csv(&time_lanczos_vec, "./python_visualization/lanczos/time_lanczos_vec.csv");

}

fn orchestrator() {
    // Set the number of Arnoldi iterations
    let n = 250; // The number of iterations give the number of orthonormal columns // 250

    // Define your matrix here
    let A = DMatrix::new_random(n, n);

    let A_H = A.clone()+A.transpose();

    write_matrix_to_csv(&A, &format!("./python_visualization/A.csv"));
    write_matrix_to_csv(&A_H, &format!("./python_visualization/A_H.csv"));

    // Choose a random initial vector and normalize it
    let mut r0 = DVector::new_random(n);
    r0.normalize_mut();

    let k_max = 200; // 200

    let k_vector: Vec<usize> = (2.=k_max).collect(); // k = Grade of the vector b

    let mut orthogonality_loss_gs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_cgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_mgs_H_vec: Vec<f64> = Vec::new();
    let mut orthogonality_loss_lanczos_vec: Vec<f64> = Vec::new();

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

    println!("\nk: {}", k_max);
    println!("\nn: {}", n);
    println!("\nGS iteration orthogonality loss:");
    println!("{:?}", orthogonality_loss_gs_vec.clone());

    println!("\nArnoldi Iteration using Classical GS orthogonality loss:");
    println!("{:?}", orthogonality_loss_cgs_vec.clone());

    println!("\nArnoldi Iteration using Modified GS orthogonality loss:");
    println!("{:?}", orthogonality_loss_mgs_vec);

    println!("\nLanczos Iteration orthogonality loss:");
    println!("{:?}", orthogonality_loss_lanczos_vec);

    // Compute and write averaged execution time: CAUTION! Long execution time
    compute_time(&A, &A_H,r0, k_max);

}



fn main() {
    orchestrator()
}