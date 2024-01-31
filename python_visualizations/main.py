import csv
import numpy as np
import matplotlib.pyplot as plt

def read_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Filter out empty strings and convert to float
            row = [float(value) if value != '' else 0.0 for value in row]
            matrix.append(row)
    return np.array(matrix)

def plot_sparse_matrix(matrix, output_file, label):
    plt.spy(matrix, markersize=5, precision=0.1, label=label)
    plt.savefig(output_file)
    plt.show()

def plot_sparse_matrices(vector):
    counter = 0
    for file in vector:
        matrix = read_matrix_from_csv(file)
        if counter == 0:
            plot_sparse_matrix(matrix, 'sparse_matrix_gs.png', label="Grand Schmidt")
        elif counter == 1:
            plot_sparse_matrix(matrix, 'sparse_matrix_cgs.png', label="Arnoldi CGS")
        elif counter == 2:
            plot_sparse_matrix(matrix, 'sparse_matrix_mgs.png', label="Arnoldi MGS")
        else:
            plot_sparse_matrix(matrix, 'sparse_matrix_lanczos.png', label="Lanczos")
        counter+=1

def read_vector_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        vector = [float(row[0]) for row in reader]
    return vector

def plot_and_save_orthogonality(vectors, output_file):
    
    counter = 0
    for vector in vectors:
        x_vector = list(range(2, 2 + len(vector)))
        if counter == 0:
            plt.plot(x_vector, vector, label='GS')
        elif counter == 1:
            plt.plot(x_vector, vector, label='Arnoldi CGS')
        elif counter == 2:
            plt.plot(x_vector, vector, label='Arnoldi MGS')
        counter+=1
    
    plt.xlabel('k')
    plt.ylabel('Orthogonality loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

def plot_and_save_orthogonality_H(vectors, output_file):
    
    counter = 0
    for vector in vectors:
        x_vector = list(range(2, 2 + len(vector)))
        if counter == 0:
            plt.plot(x_vector, vector, label='Arnoldi MGS')
        elif counter == 1:
            plt.plot(x_vector, vector, label='Lanczos')
        counter+=1
    
    plt.xlabel('k')
    plt.ylabel('Orthogonality loss for Hermitian matrix')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_file)
    plt.show()

def plot_and_save_time(vectors, output_file):

    counter = 0
    for vector in vectors:
        x_vector = list(range(2, 2 + len(vector)))
        if counter == 0:
            plt.plot(x_vector, vector, label='GS')
        elif counter == 1:
            plt.plot(x_vector, vector, label='Arnoldi CGS')
        elif counter == 2:
            plt.plot(x_vector, vector, label='Arnoldi MGS')
        else:
            plt.plot(x_vector, vector, label='Lanczos')
        counter+=1

    plt.xlabel('k')
    plt.ylabel('Execution time')
    plt.legend()
    plt.savefig(output_file)  # Save as PNG file
    plt.show()

def plot_and_save_time_H(vectors, output_file):

    counter = 0
    for vector in vectors:
        x_vector = list(range(2, 2 + len(vector)))
        if counter == 0:
            plt.plot(x_vector, vector, label='Arnoldi MGS')
        elif counter == 1:
            plt.plot(x_vector, vector, label='Lanczos')
        counter+=1

    plt.xlabel('k')
    plt.ylabel('Execution time for Hermitian matrix')
    plt.legend()
    plt.savefig(output_file)  # Save as PNG file
    plt.show()

def main():
    orthogonality_loss_gs_file = 'gs/orthogonality_loss_gs_vec.csv'  
    orthogonality_loss_cgs_file = 'cgs/orthogonality_loss_cgs_vec.csv'  
    orthogonality_loss_mgs_file = 'mgs/orthogonality_loss_mgs_vec.csv'
    orthogonality_loss_mgs_H_file = 'mgs/orthogonality_loss_mgs_H_vec.csv'
    orthogonality_loss_lanczos_file = 'lanczos/orthogonality_loss_lanczos_vec.csv'

    time_gs_file = 'gs/time_gs_vec.csv' 
    time_cgs_file = 'cgs/time_cgs_vec.csv'  
    time_mgs_file = 'mgs/time_mgs_vec.csv'
    time_mgs_H_file = 'mgs/time_mgs_H_vec.csv'
    time_lanczos_file = 'lanczos/time_lanczos_vec.csv'

    r_path_gs = 'gs/R/R_gs_45.csv'
    h_path_cgs = 'cgs/H/H_cgs_45.csv' 
    h_path_mgs = 'mgs/H/H_mgs_45.csv'
    t_path_lanczos = 'lanczos/T/T_lanczos_45.csv'

    orthogonality_loss_gs_vec = read_vector_from_csv(orthogonality_loss_gs_file)
    orthogonality_loss_cgs_vec = read_vector_from_csv(orthogonality_loss_cgs_file)
    orthogonality_loss_mgs_vec = read_vector_from_csv(orthogonality_loss_mgs_file)
    orthogonality_loss_mgs_H_vec = read_vector_from_csv(orthogonality_loss_mgs_H_file)
    orthogonality_loss_lanczos_vec = read_vector_from_csv(orthogonality_loss_lanczos_file)

    time_gs_vec = read_vector_from_csv(time_gs_file)
    time_cgs_vec = read_vector_from_csv(time_cgs_file)
    time_mgs_vec = read_vector_from_csv(time_mgs_file)
    time_mgs_H_vec = read_vector_from_csv(time_mgs_H_file)
    time_lanczos_vec = read_vector_from_csv(time_lanczos_file)

    orthogonality_vec = [orthogonality_loss_gs_vec, orthogonality_loss_cgs_vec, orthogonality_loss_mgs_vec]
    orthogonality_vec_H = [orthogonality_loss_mgs_H_vec, orthogonality_loss_lanczos_vec]
    time_vec = [time_gs_vec, time_cgs_vec, time_mgs_vec]
    time_vec_H = [time_mgs_H_vec, time_lanczos_vec]
    
    plot_and_save_orthogonality(orthogonality_vec, 'orthogonality.png')
    plot_and_save_orthogonality_H(orthogonality_vec_H, 'orthogonality_H.png')
    plot_and_save_time(time_vec, 'time.png')
    plot_and_save_time_H(time_vec_H, 'time_H.png')

    path_vec = [r_path_gs, h_path_cgs, h_path_mgs, t_path_lanczos]
    plot_sparse_matrices(path_vec)

if __name__ == "__main__":
    main()
