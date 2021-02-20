import torch
from utils.timer import Timer

cpu_timer = Timer()
gpu_timer = Timer()
trials = 10
num_corr = 400

Covariances = torch.abs(torch.rand([num_corr, 3, 3]))
for i in range(trials):
    cpu_timer.tic()
    u, s, vt = torch.svd(Covariances)
    cpu_timer.toc()
print(f"CPU Average Time for {num_corr} SVD: {cpu_timer.avg:6f}s")

cpu_timer.reset()
Covariances = torch.abs(torch.rand([num_corr, 3, 3]))
for i in range(trials):
    cpu_timer.tic()
    for j in range(num_corr):
        u, s, vt = torch.svd(Covariances[j])
    cpu_timer.toc()
print(f"Serial CPU Average Time for {num_corr} SVD: {cpu_timer.avg:6f}s")

Covariances = torch.abs(torch.rand([num_corr, 3, 3])).cuda()
for i in range(trials):
    gpu_timer.tic()
    u, s, vt = torch.svd(Covariances)
    gpu_timer.toc()
print(f"GPU Average Time for {num_corr} SVD: {gpu_timer.avg:6f}s")

## https://github.com/KinglittleQ/torch-batch-svd.git
# import torch
# from torch_batch_svd import svd
# import time
#
# def torch_svd_cpu(A):
#     u,s,v = torch.svd(A.cpu())
#     u,s,v = u.cuda(), s.cuda(), v.cuda()
#     return u,s,v
#
# for i in [100, 1000, 10000, 100000, 1000000]:
#     print(f"Number of 3*3 SVD={i}")
#     print("-"*20)
#     A = torch.rand(i, 3, 3).cuda()
#     %time u,s,v = svd(A)
#     %time uu,ss,vv = torch_svd_cpu(A)
#     %time uuu, sss, vvv = torch.svd(A)
