use nalgebra::{DMatrix, DVector, Vector};
use std::io::prelude::*;
use std::path::Path;
use std::fs::File;
use csv::WriterBuilder;
use std::error::Error;



pub fn write_matrix_to_csv(matrix: &DMatrix<f64>, filename: &str) -> Result<(), std::io::Error> {
    /**
    Writes a matrix into csv format.

    Args:
    - matrix (DMatrix<f64>): The matrix to write.
    - filename (str): The path and file name.

    Returns:
    - None
    */

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

pub fn write_matrices_to_csv(V_gs: &DMatrix<f64>, R_gs: &DMatrix<f64>, V_cgs: &DMatrix<f64>, H_cgs: &DMatrix<f64>, V_mgs: &DMatrix<f64>, H_mgs: &DMatrix<f64>, V_lanczos: &DMatrix<f64>, T_lanczos: &DMatrix<f64>, k: usize) -> Result<(), std::io::Error> {
    /**
    Writes matrices to CSV files with specific naming conventions.

    Args:
    - V_gs (DMatrix<f64>): Orthogonal matrix for Gram-Schmidt algorithm.
    - R_gs (DMatrix<f64>): Triangular matrix for Gram-Schmidt algorithm.
    - V_cgs (DMatrix<f64>): Orthogonal matrix for Classical Gram-Schmidt algorithm.
    - H_cgs (DMatrix<f64>): Hessenberg matrix for Classical Gram-Schmidt algorithm.
    - V_mgs (DMatrix<f64>): Orthogonal matrix for Modified Gram-Schmidt algorithm.
    - H_mgs (DMatrix<f64>): Hessenberg matrix for Modified Gram-Schmidt algorithm.
    - V_lanczos (DMatrix<f64>): Orthogonal matrix for Lanczos algorithm.
    - T_lanczos (DMatrix<f64>): Tridiagonal matrix for Lanczos algorithm.
    - k (usize): Dimension of the subspace.

    Returns:
    - Result<(), std::io::Error>
    */
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

pub fn write_vector_to_csv(data: &Vec<f64>, file_path: &str) -> Result<(), Box<dyn Error>> {
    /**
    Writes a vector to a CSV file.

    Args:
    - data (Vec<f64>): Vector of data to write to the CSV file.
    - file_path (str): The path to the CSV file.

    Returns:
    - Result<(), Box<dyn Error>>
    */

    let file = File::create(file_path)?;
    let mut writer = WriterBuilder::new().from_writer(file);

    for value in data {
        writer.write_record(&[value.to_string()])?;
    }

    writer.flush()?;
    Ok(())
}

pub fn write_orthogonality_vectors_to_csv(orthogonality_loss_gs_vec: &Vec<f64>, orthogonality_loss_cgs_vec: &Vec<f64>, orthogonality_loss_mgs_vec: &Vec<f64>, orthogonality_loss_mgs_H_vec: &Vec<f64>, orthogonality_loss_lanczos_vec: &Vec<f64>) -> Result<(), std::io::Error> {
    /**
    Writes orthogonality loss vectors to CSV files with specific naming conventions.

    Args:
    - orthogonality_loss_gs_vec (Vec<f64>): Orthogonality loss vector for Gram-Schmidt algorithm.
    - orthogonality_loss_cgs_vec (Vec<f64>): Orthogonality loss vector for Classical Gram-Schmidt algorithm.
    - orthogonality_loss_mgs_vec (Vec<f64>): Orthogonality loss vector for Modified Gram-Schmidt algorithm.
    - orthogonality_loss_mgs_H_vec (Vec<f64>): Orthogonality loss vector for Modified Gram-Schmidt algorithm (Hermitian matrix).
    - orthogonality_loss_lanczos_vec (Vec<f64>): Orthogonality loss vector for Lanczos algorithm.

    Returns:
    - Result<(), std::io::Error>
    */
    
    write_vector_to_csv(orthogonality_loss_gs_vec, "./experiment_results/gs/orthogonality_loss_gs_vec.csv");
    write_vector_to_csv(orthogonality_loss_cgs_vec, "./experiment_results/cgs/orthogonality_loss_cgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_vec, "./experiment_results/mgs/orthogonality_loss_mgs_vec.csv");
    write_vector_to_csv(orthogonality_loss_mgs_H_vec, "./experiment_results/mgs/orthogonality_loss_mgs_H_vec.csv");
    write_vector_to_csv(orthogonality_loss_lanczos_vec, "./experiment_results/lanczos/orthogonality_loss_lanczos_vec.csv");
    Ok(())
}
