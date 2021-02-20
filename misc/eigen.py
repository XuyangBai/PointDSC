import numpy as np
import torch
import time

np.set_printoptions(precision=3)
np.random.seed(123)


def check_eigen_decompose(num_dim, max_iter):
    # random matrix (positive)
    matrix_np = np.abs(np.random.randn(num_dim,num_dim))
    # make symmetric
    matrix_np = matrix_np + matrix_np.T
    matrix_torch = torch.autograd.Variable(torch.from_numpy(matrix_np), requires_grad=True)
    matrix_pi = torch.autograd.Variable(torch.from_numpy(matrix_np), requires_grad=True)

    
    # compute eigenvalue decompositions using numpy
    eigvals_np, eigvecs_np = np.linalg.eigh(matrix_np)
    # compute eigenvalue decompositions using PyTorch
    stime = time.time()
    eigvals_torch, eigvecs_torch = torch.symeig(matrix_torch, eigenvectors=True, upper=True)
    etime = time.time()
    time_eig = etime - stime
    leading_eig_torch = eigvecs_torch[:,-1]
    # print(f"Torch Leading EigenVector {etime-stime:.10f}s:", eigvecs_torch[:,-1].data.numpy())
    assert np.allclose(eigvals_np, eigvals_torch.data.numpy())

    # Power Iteration Algorithm 
    stime = time.time()
    solution = torch.ones_like(matrix_pi[:, 0:1])
    stop_iteration = max_iter
    for i in range(max_iter):
        last_solution = solution
        solution = matrix_pi @ solution
        solution = solution / torch.norm(solution)
        error = float(torch.norm(leading_eig_torch - solution.squeeze(), p=2))
        error2 = float(torch.norm(leading_eig_torch + solution.squeeze(), p=2))
        if torch.norm(solution - last_solution) < 1e-10:
            stop_iteration = i
            etime = time.time()
            time_pi = etime - stime 
            assert np.abs(error) < 1e-6 or np.abs(error2) < 1e-6               
            print(f"Converge at iter-{stop_iteration}")
            break
    if i == max_iter-1:
        etime = time.time()
        time_pi = etime - stime
    solution = solution.squeeze(-1)

    
    ## compute derivative of first eigenvalue with respect to the matrix
    # # Np gradient, see "On differentiating eigenvalues and eigenvectors" by Jan R. Magnus
    # grad_analytic = np.outer(eigvecs_np[:, 0], eigvecs_np[:, 0])
    # # PyTorch gradient
    # eigvals_torch[0].backward()
    # grad_torch = matrix_torch.grad.numpy()
    # assert np.allclose(grad_analytic, grad_torch)

    # comparing the gradient of Power iteration and torch.symeig
    leading_eig_torch[0].backward()
    grad_eig = matrix_torch.grad.numpy()
    solution[0].backward()
    grad_pi = matrix_pi.grad.numpy()
    # print(np.linalg.norm(grad_eig - grad_pi))
    # assert np.allclose(grad_eig, grad_pi)
    return time_eig, time_pi, stop_iteration, float(i!=max_iter-1)
    

if __name__ == '__main__':
    time_eig_list = []
    time_pi_list = []
    stop_iteration_list = []
    converegence_list = []
    for i in range(200, 400):
        time_eig, time_pi, stop_iteration, flag = check_eigen_decompose(i, max_iter=500)
        time_eig_list.append(time_eig)
        time_pi_list.append(time_pi)
        stop_iteration_list.append(stop_iteration)
        converegence_list.append(flag)
    print(f"Eig mean time: {np.mean(time_eig_list):.6f}s")
    print(f"PI mean time: {np.mean(time_pi_list):.6f}s")
    print(f"PI mean iter: {np.mean(stop_iteration_list):.6f}")
    print(f"PI convergance ratio: {np.mean(converegence_list):.2f}")