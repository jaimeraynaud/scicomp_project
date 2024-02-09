# Python code to visualize our experimental results.


import os
from visualization import plot_orthogonality, plot_execution_time, plot_sparse_matrix
from data_handling import read_vector_from_csv, read_vector_from_row_csv
from matplotlib import colors as mcolors



colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

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

    time_gs_file_matlab = 'experiment_results/gs/time_gs_vec_matlab.csv'
    time_cgs_file_matlab = 'experiment_results/cgs/time_cgs_vec_matlab.csv'
    time_mgs_file_matlab = 'experiment_results/mgs/time_mgs_vec_matlab.csv'
    time_mgs_H_file_matlab = 'experiment_results/mgs/time_mgs_H_vec_matlab.csv'
    time_lanczos_file_matlab = 'experiment_results/lanczos/time_lanczos_vec_matlab.csv'

    r_path_gs = 'experiment_results/gs/R/R_gs_50.csv'
    h_path_cgs = 'experiment_results/cgs/H/H_cgs_50.csv' 
    h_path_mgs = 'experiment_results/mgs/H/H_mgs_50.csv'
    t_path_lanczos = 'experiment_results/lanczos/T/T_lanczos_50.csv'

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

    time_gs_vec_matlab = read_vector_from_row_csv(time_gs_file_matlab)
    time_cgs_vec_matlab = read_vector_from_row_csv(time_cgs_file_matlab)
    time_mgs_vec_matlab = read_vector_from_row_csv(time_mgs_file_matlab)
    time_mgs_H_vec_matlab = read_vector_from_row_csv(time_mgs_H_file_matlab)
    time_lanczos_vec_matlab = read_vector_from_row_csv(time_lanczos_file_matlab)

    orthogonality_vec = [orthogonality_loss_gs_vec, orthogonality_loss_cgs_vec, orthogonality_loss_mgs_vec]
    orthogonality_vec_H = [orthogonality_loss_mgs_H_vec, orthogonality_loss_lanczos_vec]
    time_vec = [time_gs_vec, time_cgs_vec, time_mgs_vec, time_lanczos_vec]
    time_vec_H = [time_mgs_H_vec, time_lanczos_vec]
    time_vec_matlab = [time_gs_vec_matlab, time_cgs_vec_matlab, time_mgs_vec_matlab]
    time_vec_H_comp = [time_mgs_H_vec, time_lanczos_vec, time_mgs_H_vec_matlab, time_lanczos_vec_matlab]

    colors = [colors_dict["r"], colors_dict["darkorange"], colors_dict["lawngreen"], colors_dict["deepskyblue"]]
    
    plot_orthogonality(orthogonality_vec, 'orthogonality_plot.png', ['Grand-Schmidt (GS)', 'Arnoldi CGS', 'Arnoldi MGS'], colors)
    plot_orthogonality(orthogonality_vec[1:3], 'orthogonality_plot_2.png', ['Arnoldi CGS', 'Arnoldi MGS'], colors[1:3])
    plot_orthogonality(orthogonality_vec_H, 'orthogonality_H_plot.png', ['Arnoldi MGS', 'Lanczos'], colors[2:])
    plot_execution_time(time_vec, 'execution_time_plot.png', ['Grand-Schmidt (GS)', 'Arnoldi CGS', 'Arnoldi MGS'], colors)
    plot_execution_time(time_vec_H, 'execution_time_H_plot.png', ['Arnoldi MGS', 'Lanczos'], colors[2:])
    plot_execution_time(time_vec_matlab, 'execution_time_matlab_plot.png', ['Grand-Schmidt (GS) (MATLAB)', 'Arnoldi CGS (MATLAB)', 'Arnoldi MGS (MATLAB)'], colors, matlab=1)
    plot_execution_time(time_vec_H_comp, 'execution_time_H_comp_plot.png', ['Arnoldi MGS (Rust)', 'Lanczos (Rust)', 'Arnoldi MGS (MATLAB)', 'Lanczos (MATLAB)'], colors[2:]+colors[2:], matlab=2)

    plot_sparse_matrix(r_path_gs, 'Grand-Schmidt (GS)', colors_dict["dodgerblue"])
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
    #delete_files_in_folder("images")

    main()

