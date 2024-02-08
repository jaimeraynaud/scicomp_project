import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from data_handling import read_vector_from_csv, read_matrix_from_csv

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
    plt.savefig(f'images/sparse_matrix_{label.replace(" ", "_").lower()}.png')
    plt.show()
