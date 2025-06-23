import torch

n_f = 10000 #[5000]
lrate = 0.005
tryerr = 2*10**(-3)
u0_t = 0
v0_t = 0

Nm = 1
Nx, Nt_all= 257, 101
if(Nt_all % Nm !=0):
    Nt = int(Nt_all/Nm) + 1
else:
    Nt = int(Nt_all/Nm)# + 1
tint = Nt -1

N0 = 256
Nb = 100

tf_iter1= [20000]
newton_iter1= [40000]
max_retry=1

num_layer=4
width=48
layer_sizes=[2]
for i in range(num_layer):
    layer_sizes.append(width)
layer_sizes.append(2)

# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print("wrong device")

