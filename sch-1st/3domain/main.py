
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import scipy.io
from pyDOE import lhs
import pickle
import os
from globalf import *
from ESA_PINN_PT import ESA_PINN_PT
#from SAPINN2 import*
import torch.autograd as autograd

path = './'
file_name = ['Data_flie','figures']#, 'saved_weights']

for i in range(len(file_name)):
    file = file_name[i]
    isExists = os.path.exists(path+str(file))
    if not isExists:
        os.makedirs(path+str(file))
        print("%s is not Exists"%file)
    else:
        print("%s is Exists"%file)
        continue

#torch.manual_seed(1234)
#np.random.seed(1234)

def loss(sapinn:ESA_PINN_PT, x_f_batch, t_f_batch, x0, t0, u0, v0, u0_t, v0_t, x_lb, t_lb, x_ub, t_ub, u_lb, u_ub, v_lb, v_ub, col_weights, u_weights, u_lb_weights, u_ub_weights, col_v_weights, v_weights, v_lb_weights, v_ub_weights):
    u0_pred,v0_pred, _,_ = sapinn.uv_model(x0,t0)
    u_lb_pred,v_lb_pred,_, _ = sapinn.uv_model(x_lb, t_lb)
    u_ub_pred,v_ub_pred,_, _ = sapinn.uv_model(x_ub, t_ub)
    f_u_pred,f_v_pred = f_model(sapinn, x_f_batch, t_f_batch)

    mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)+ torch.mean((v_weights * (v0 - v0_pred))**2)
    mse_b_u = torch.mean((u_lb_weights*(u_lb - u_lb_pred))**2) + torch.mean((u_ub_weights*(u_ub - u_ub_pred))**2)
    mse_b_v = torch.mean((v_lb_weights*(v_lb - v_lb_pred))**2) + torch.mean((v_ub_weights*(v_ub - v_ub_pred))**2)
    mse_f_u = torch.mean((col_weights * f_u_pred)**2) + torch.mean((col_v_weights * f_v_pred)**2)
    
    return mse_0_u + mse_b_u + mse_b_v + mse_f_u, mse_0_u, mse_b_u + mse_b_v, mse_f_u

def ensure_grad(tensors):
    for tensor in tensors:
        if not tensor.requires_grad:
            tensor.requires_grad = True

def f_model(sapinn:ESA_PINN_PT, x, t):

    u, v, _, _ = sapinn.uv_model(x,t)
    u_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_t = autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    
    f_v=u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_u=v_t - 0.5*u_xx - (u**2 + v**2)*u
    
    return f_u,f_v


for id_t in range(Nm):
    model=ESA_PINN_PT("data-x2t2.mat",layer_sizes,tf_iter1[id_t],newton_iter1[id_t],f_model=f_model,Loss=loss,N_f= n_f, id_t= id_t, u0_t= u0_t,v0_t=v0_t)

    model.fit()
    model.fit_lbfgs()
    
    # Save the model
    torch.save(model.model.state_dict(), model.checkPointPath + "/final-%d.pth"%id_t)
    
    error_u_value, error_v_value = model.error_u()
    print('Error u: %e' % (error_u_value))
    print('Error v: %e' % (error_v_value))

    u_pred,v_pred, f_u_pred,f_v_pred = model.predict()#u_model, X_star)
    
    U3_pred = u_pred.reshape((Nt, Nx)).T
    V3_pred = v_pred.reshape((Nt, Nx)).T

    f_U3_pred = f_u_pred.reshape((Nt, Nx)).T
    f_V3_pred = f_v_pred.reshape((Nt, Nx)).T
    
    u0_t= U3_pred[:,-1]
    v0_t= V3_pred[:,-1]

    utmp = U3_pred
    vtmp = V3_pred
    f_utmp = f_U3_pred
    f_vtmp = f_V3_pred
 
 
    if(id_t>0):
       comb_u_pred = np.concatenate((comb_u_pred[:,:-1], utmp), axis=1)
       comb_v_pred = np.concatenate((comb_v_pred[:,:-1], vtmp), axis=1)
       comb_f_u_pred = np.concatenate((comb_f_u_pred[:,:-1], f_utmp), axis=1)
       comb_f_v_pred = np.concatenate((comb_f_v_pred[:,:-1], f_vtmp), axis=1)
 
       Exact_u_i = model.Exact_u[:, :(tint*(id_t+1) +1)]
       Exact_v_i = model.Exact_v[:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_v  = np.linalg.norm((Exact_v_i - comb_v_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)
       perror_vEx = np.linalg.norm(Exact_v_i.flatten(),2)
 
       error_u = perror_u/perror_uEx
       error_v = perror_v/perror_vEx
       print('Error u: %e' % (error_u))
       print('Error v: %e' % (error_v))

       if(id_t==Nm-1):
           Exact_u_i = model.Exact_u
           Exact_v_i = model.Exact_v
           H_Exact = np.sqrt(Exact_u_i**2 + Exact_v_i**2)
           H_pred = np.sqrt(comb_u_pred**2 + comb_v_pred**2)
           perror_h  = np.linalg.norm((H_Exact - H_pred).flatten(),2)
           perror_hEx = np.linalg.norm(H_Exact.flatten(),2)
           error_h = perror_h/perror_hEx
           print('Error h: %e' % (error_h))

 
    else:
       comb_u_pred = U3_pred
       comb_v_pred = V3_pred
       comb_f_u_pred = f_U3_pred
       comb_f_v_pred = f_V3_pred
 
       Exact_u_i = model.Exact_u[:, :(tint*(id_t+1) +1)]
       Exact_v_i = model.Exact_v[:, :(tint*(id_t+1) +1)]
       perror_u  = np.linalg.norm((Exact_u_i - comb_u_pred).flatten(),2)
       perror_v  = np.linalg.norm((Exact_v_i - comb_v_pred).flatten(),2)
       perror_uEx = np.linalg.norm(Exact_u_i.flatten(),2)
       perror_vEx = np.linalg.norm(Exact_v_i.flatten(),2)
 
       error_u = perror_u/perror_uEx
       error_v = perror_v/perror_vEx
       print('Error u: %e' % (error_u))
       print('Error v: %e' % (error_v))
 
 #   if (ferru < tryerr and ferrv < tryerr):
 #   	break
 #   else:
 #   	pass
 
    print('u_pred.shape after= ', comb_u_pred.shape)

pickle_file1 = open('Data_flie/comb_u_pred.pkl', 'wb')
pickle.dump(comb_u_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_u_pred.pkl', 'wb')
pickle.dump(comb_f_u_pred, pickle_file2)
pickle_file2.close()

pickle_file1 = open('Data_flie/comb_v_pred.pkl', 'wb')
pickle.dump(comb_v_pred, pickle_file1)
pickle_file1.close()
pickle_file2 = open('Data_flie/comb_f_v_pred.pkl', 'wb')
pickle.dump(comb_f_v_pred, pickle_file2)
pickle_file2.close()

