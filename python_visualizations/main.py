import os

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

def read_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Filter out empty strings and convert to float
            row = [float(value) if value != '' else 0.0 for value in row]
            matrix.append(row)
    return np.array(matrix)

def read_vector_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        vector = [float(row[0]) for row in reader]
    return vector

def read_vector_from_row_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # Assuming there is only one row in the file
        row = next(reader)
        vector = [float(value) for value in row]
    return vector

def plot_orthogonality(vectors, output_file, labels, colors):
    plt.figure(figsize=(10, 6))
    
    for vector, label, color in zip(vectors, labels, colors):
        x_vector = list(range(2, 2 + len(vector)))
        plt.plot(x_vector, vector, label=label, marker='o', markersize=3, color=color)

    plt.xlabel('k', fontsize=12)
    plt.ylabel('Orthogonality loss', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("images/"+output_file)
    plt.show()

def plot_execution_time(vectors, output_file, labels, colors, matlab=0):
    plt.figure(figsize=(10, 6))
    
    vector_index=0

    for vector, label, color in zip(vectors, labels, colors):
        x_vector = list(range(2, 2 + len(vector)))
        line_style = '--' if matlab==1 or (matlab==2 and (vector_index == 2 or vector_index==3)) else '-'
        plt.plot(x_vector, vector, label=label, color=color, linestyle=line_style)
        vector_index+=1

    plt.xlabel('k', fontsize=12)
    plt.ylabel('Execution time (s)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("images/"+output_file)
    plt.show()

def plot_sparse_matrix(path, label, color):
    plt.figure(figsize=(10, 6))
    matrix = read_matrix_from_csv(path)
    plt.spy(matrix, markersize=3, precision=0.1, label=label, color=color)
    plt.tight_layout()
    plt.savefig("images/"+f'sparse_matrix_{label.replace(" ", "_").lower()}.png')
    plt.show()

def main():
    
    print("\n===============================================================================================================================\n")
    print("Python visualizations of experimental results")
    print("\n1. Matrix structures")
    print("\n2. Orthogonality loss")
    print("\n3. Runtime")
    print("\n===============================================================================================================================\n")


    orthogonality_loss_gs_file = 'experiment_results/gs/orthogonality_loss_gs_vec.csv'  
    orthogonality_loss_cgs_file = 'experiment_results/cgs/orthogonality_loss_cgs_vec.csv'  
    orthogonality_loss_mgs_file = 'experiment_results/mgs/orthogonality_loss_mgs_vec.csv'
    orthogonality_loss_mgs_H_file = 'experiment_results/mgs/orthogonality_loss_mgs_H_vec.csv'
    orthogonality_loss_lanczos_file = 'experiment_results/lanczos/orthogonality_loss_lanczos_vec.csv'

    time_gs_file = 'experiment_results/gs/time_gs_vec.csv' 
    time_cgs_file = 'experiment_results/cgs/time_cgs_vec.csv'  
    time_mgs_file = 'experiment_results/mgs/time_mgs_vec.csv'
    time_mgs_H_file = 'experiment_results/mgs/time_mgs_H_vec.csv'
    time_lanczos_file = 'experiment_results/lanczos/time_lanczos_vec.csv'
    time_mgs_H_file_matlab = 'experiment_results/mgs/time_mgs_H_vec_matlab.csv'
    time_lanczos_file_matlab = 'experiment_results/lanczos/time_lanczos_vec_matlab.csv'

    r_path_gs = 'experiment_results/gs/R/R_gs_12.csv'
    h_path_cgs = 'experiment_results/cgs/H/H_cgs_12.csv' 
    h_path_mgs = 'experiment_results/mgs/H/H_mgs_12.csv'
    t_path_lanczos = 'experiment_results/lanczos/T/T_lanczos_12.csv'

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
    time_mgs_H_vec_matlab = read_vector_from_row_csv(time_mgs_H_file_matlab)
    time_lanczos_vec_matlab = read_vector_from_row_csv(time_lanczos_file_matlab)

    orthogonality_vec = [orthogonality_loss_gs_vec, orthogonality_loss_cgs_vec, orthogonality_loss_mgs_vec]
    orthogonality_vec_H = [orthogonality_loss_mgs_H_vec, orthogonality_loss_lanczos_vec]
    time_vec = [time_gs_vec, time_cgs_vec, time_mgs_vec, time_lanczos_vec]
    time_vec_H = [time_mgs_H_vec, time_lanczos_vec]
    time_vec_H_matlab = [time_mgs_H_vec_matlab, time_lanczos_vec_matlab]
    time_vec_H_comp = [time_mgs_H_vec, time_lanczos_vec, time_mgs_H_vec_matlab, time_lanczos_vec_matlab]

    colors = [colors_dict["r"], colors_dict["darkorange"], colors_dict["lawngreen"], colors_dict["deepskyblue"]]
    
    plot_orthogonality(orthogonality_vec, 'orthogonality_plot.png', ['Classical Grand Schmidt (GS)', 'Arnoldi CGS', 'Arnoldi MGS'], colors)
    plot_orthogonality(orthogonality_vec[1:3], 'orthogonality_plot_2.png', ['Arnoldi CGS', 'Arnoldi MGS'], colors[1:3])
    plot_orthogonality(orthogonality_vec_H, 'orthogonality_H_plot.png', ['Arnoldi MGS', 'Lanczos'], colors[2:])
    plot_execution_time(time_vec, 'execution_time_plot.png', ['Classical Grand Schmidt (GS)', 'Arnoldi CGS', 'Arnoldi MGS'], colors)
    plot_execution_time(time_vec_H, 'execution_time_H_plot.png', ['Arnoldi MGS', 'Lanczos'], colors[2:])
    plot_execution_time(time_vec_H_matlab, 'execution_time_H_matlab_plot.png', ['Arnoldi MGS (MatLab)', 'Lanczos (MatLab)'], colors[2:], matlab=1)
    plot_execution_time(time_vec_H_comp, 'execution_time_H_comp_plot.png', ['Arnoldi MGS (Rust)', 'Lanczos (Rust)', 'Arnoldi MGS (MatLab)', 'Lanczos (MatLab)'], colors[2:]+colors[2:], matlab=2)

    plot_sparse_matrix(r_path_gs, 'Classical Grand Schmidt (GS)', colors_dict["dodgerblue"])
    plot_sparse_matrix(h_path_cgs, 'Arnoldi CGS', colors_dict["dodgerblue"])
    plot_sparse_matrix(h_path_mgs, 'Arnoldi MGS', colors_dict["dodgerblue"])
    plot_sparse_matrix(t_path_lanczos, 'Lanczos', colors_dict["dodgerblue"])

def delete_files_in_folder(folder_path):
    """
    Deletes all files in a given nested folder.

    Args:
    - folder_path (str): The path to the folder.

    Returns:
    - None
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

if __name__ == "__main__":
    #delete_files_in_folder("experiment_results")
    main()

