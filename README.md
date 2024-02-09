# Scientific Computing Final Project Setup and Execution Guide

Welcome to our project. This document outlines the steps necessary to execute our Rust code, run the Matlab scripts for file generation, and how to visualize the data using a Python script. Moreover, it furnishes the instructions to modify the three parameters (`n`, `k_max` and `num_iterations`) on the different files to test the code for different setups. 
Please follow the instructions carefully to ensure a smooth setup and execution process.

## 1. Executing Rust Code

To begin with, you will need to execute the Rust code located in the `src` folder. Here is how you can achieve that:

### Steps:

1. **Navigate to the project Folder `scicomp_project`:**
   - **Option 1:** Open your terminal. Use the `cd` command to change directories until you are inside the `scicomp_project` folder of our project.
   - **Option 2:** Alternatively, navigate to the `scicomp_project` folder, then right-click in the folder and select the option to open a terminal or command prompt window directly in that location.

2. **Run the Rust Code:**
   - In the terminal window that is now open in the `scicomp_project` folder, type the following command and press Enter:
     ```shell
     cargo run
     ```
   - Wait for the execution to complete. This will compile and run the Rust application as configured in the project.

## 2. Running Matlab Codes

After you have successfully executed the Rust code, the next step is to run the Matlab script for file generation.

### Steps:

Open Matlab and execute the Matlab script file provided in the `matlab_version` folder. If you do not have Matlab installed on your computer, you can use Matlab Online to run the script.

**ACHTUNG!**: be aware to have the latest version of MATLAB. In case the code gives an error about the size of the matrices, it means you have an old version of the software. If this happens uncomment lines 19 and 22.

## 3. Visualizing Data with Python

To visualize the data generated by the Matlab script, run the provided Python script from the `python_visualizations` folder.

### Steps:

Run the Python script provided in the `python_visualizations` folder using an IDE of your choice and make sure it is being run from the correct directory.

## (extra) How to modify the parameters

In case you would like to reproduce our results or test the code for different parameters, here is what you should modify and on which files. At first, the three essential parameters for the execution are
   - `n`: size of the initial random matrix
   - `k_max`: dimension of the Krylov subspace, always less than `n`
   - `num_iterations`: number of iterations on which measure the average runtime execution

### Steps:

1. **Rust code:**
      - inside `scicomp_project`, access `src` folder and open the file `main.rs`
      - go to line 43 and 44, here you can change `n` and `k_max`. To reproduce our results set respectively 250 and 200 as values
      - still inside `src` folder, open the file `algo_analysis.rs`
      - go to line 23, here you can change `num_iterations`. To reproduce our results set 10 as value

2. **MATLAB code:**
      - inside `scicomp_project`, access `matlab_version` folder and open the file `main.m`
      - go to line 10 and 11, here you can change `n` and `k_max`. To reproduce our results set respectively 250 and 200 as values
      - go to line 33, here you can change `num_iterations`. To reproduce our results set 10 as value
  
3. **Python code:**
      - inside `scicomp_project`, access `python_visualizations` folder and open the file `main.py`
      - go to line 41, 42, 43 and 44, here for each `.csv` file change the number in the end, which indicates the dimension of the matrices you want to visualize (R, H and T), this number needs to be less or equal than `k_max`. For our results, we set it as `k_max` then, for example, the file will be like that `R_gs_200.csv`

## Conclusion

By following these steps, you should be able to successfully execute the Rust code, run the Matlab scripts, and visualize the data using Python. Should you encounter any issues, please review the steps to ensure all instructions have been followed accurately.

If any error arises contact jaime.raynaud.sanchez@tu-berlin.de
