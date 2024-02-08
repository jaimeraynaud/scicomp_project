import csv
import numpy as np

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
