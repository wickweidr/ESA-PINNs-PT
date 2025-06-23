
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
import datetime

class ESA_PINN_PT:
    def __DefaultLoss(self,x_f_batch, t_f_batch,
             x0, t0, u0,u_lb,u_ub, x_lb,
             t_lb, x_ub, t_ub, col_weights, u_weights):
#             t_lb, x_ub, t_ub,SA_weight):
        u0_pred = self.u_model(torch.cat([x0, t0], 1))
        u_lb_pred, u_x_lb_pred = self.u_x_model(self.model, x_lb, t_lb)
        u_ub_pred, u_x_ub_pred = self.u_x_model(self.model, x_ub, t_ub)
        f_u_pred = self.f_model(self.model, x_f_batch, t_f_batch)
        
        mse_0_u = torch.mean((u_weights * (u0 - u0_pred))**2)
        mse_b_u = torch.mean((u_lb_pred - u_ub_pred)**2) + torch.mean((u_x_lb_pred - u_x_ub_pred)**2)
        mse_f_u = torch.mean((col_weights * f_u_pred)**2)
        
        return mse_0_u + mse_b_u + mse_f_u, mse_0_u, mse_b_u, mse_f_u

    def uv_model(self,x,t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        uv = self.model(torch.cat([x, t], dim=1))
        u=uv[:,0:1]
        v=uv[:,1:2]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u,v,u_x,v_x

    def __init__(self,mat_filename,layers:[],adam_iter:int,newton_iter:int,f_model,Loss=__DefaultLoss,lbfgs_lr=0.8,N_f=10000,id_t=0,u0_t=0,v0_t=0,checkPointPath="./checkPoint"):
        self.N_f=N_f
        self.u0_t= u0_t
        self.v0_t=v0_t
        self.id_t= id_t
        self.__Loadmat(mat_filename)
        self.layers=layers
        self.sizes_w=[]
        self.sizes_b=[]
        self.lbfgs_lr=lbfgs_lr

        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        self.col_weights = nn.Parameter(torch.full((N_f, 1), 100.0, device=device))
        self.u_weights = nn.Parameter(torch.full((self.x0.shape[0], 1), 100.0, device=device))
        self.u_lb_weights = nn.Parameter(torch.full((self.x_lb.shape[0], 1), 100.0, device=device))
        self.u_ub_weights = nn.Parameter(torch.full((self.x_ub.shape[0], 1), 100.0, device=device))
        self.col_v_weights = nn.Parameter(torch.full((N_f, 1), 100.0, device=device))
        self.v_weights = nn.Parameter(torch.full((self.x0.shape[0], 1), 100.0, device=device))
        self.v_lb_weights = nn.Parameter(torch.full((self.x_lb.shape[0], 1), 100.0, device=device))
        self.v_ub_weights = nn.Parameter(torch.full((self.x_ub.shape[0], 1), 100.0, device=device))

        class NeuralNet(nn.Module):
            def __init__(self, layer_sizes):
                super(NeuralNet, self).__init__()
                layers = []
                input_size = layer_sizes[0]
                for output_size in layer_sizes[1:-1]:
                    layers.append(nn.Linear(input_size, output_size))
                    layers.append(nn.Tanh())
                    input_size = output_size
                layers.append(nn.Linear(input_size, layer_sizes[-1]))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)

        self.model = NeuralNet(self.layers)
        print(self.model)
        self.model = self.model.cuda()

        self.Loss=Loss
        self.adam_iter=adam_iter
        self.newton_iter=newton_iter
        self.f_model=f_model
    
        self.checkPointPath = f"{checkPointPath}"
        if not os.path.exists(self.checkPointPath):
            os.makedirs(self.checkPointPath)


    def ggrad(self, model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, v0_batch, u0_t_batch, v0_t_batch, x_lb, t_lb, x_ub, t_ub):
        x_f_batch = x_f_batch.to(device).requires_grad_(True)
        t_f_batch = t_f_batch.to(device).requires_grad_(True)
        x0_batch = x0_batch.to(device).requires_grad_(True)
        t0_batch = t0_batch.to(device).requires_grad_(True)
        u0_batch = u0_batch.to(device).requires_grad_(True)
        v0_batch = v0_batch.to(device).requires_grad_(True)
        u0_t_batch = u0_t_batch.to(device).requires_grad_(True)
        v0_t_batch = v0_t_batch.to(device).requires_grad_(True)
        x_lb = x_lb.to(device).requires_grad_(True)
        t_lb = t_lb.to(device).requires_grad_(True)
        x_ub = x_ub.to(device).requires_grad_(True)
        t_ub = t_ub.to(device).requires_grad_(True)
    
        model.zero_grad()
    
        loss_value, mse_0, mse_b, mse_f = self.Loss(self, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, v0_batch, u0_t_batch, v0_t_batch, x_lb, t_lb, x_ub, t_ub, self.u_lb, self.u_ub, self.v_lb, self.v_ub, self.col_weights, self.u_weights, self.u_lb_weights, self.u_ub_weights, self.col_v_weights, self.v_weights, self.v_lb_weights, self.v_ub_weights)
    
        loss_value.backward(retain_graph=True)
        grads = [param.grad.clone() for param in model.parameters()]
        model.zero_grad()
        loss_value.backward(retain_graph=True)
        grads_col = self.col_weights.grad.clone()
        grads_u = self.u_weights.grad.clone()
        grads_u_lb = self.u_lb_weights.grad.clone()
        grads_u_ub = self.u_ub_weights.grad.clone()
    
        grads_col_v = self.col_v_weights.grad.clone()
        grads_v = self.v_weights.grad.clone()
        grads_v_lb = self.v_lb_weights.grad.clone()
        grads_v_ub = self.v_ub_weights.grad.clone()
    
        return loss_value.item(), mse_0.item(), mse_b.item(), mse_f.item(), grads, grads_col, grads_u, grads_u_lb, grads_u_ub, grads_col_v, grads_v, grads_v_lb, grads_v_ub


    # Define the training loop
    def fit(self):
    
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.005, betas=(0.99, 0.999))
        optimizer_col_weights = optim.Adam([self.col_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_u_weights = optim.Adam([self.u_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_u_lb_weights = optim.Adam([self.u_lb_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_u_ub_weights = optim.Adam([self.u_ub_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_col_v_weights = optim.Adam([self.col_v_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_v_weights = optim.Adam([self.v_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_v_lb_weights = optim.Adam([self.v_lb_weights], lr=0.005, betas=(0.99, 0.999))
        optimizer_v_ub_weights = optim.Adam([self.v_ub_weights], lr=0.005, betas=(0.99, 0.999))
    
        print("starting Adam training")
    
        # For mini-batch (if used)
        for epoch in range(self.adam_iter):
            for i in range(n_batches):
    
                x0_batch = torch.tensor(self.x0, dtype=torch.float32)
                t0_batch = torch.tensor(self.t0, dtype=torch.float32)
                u0_batch = torch.tensor(self.u0, dtype=torch.float32)
                v0_batch = torch.tensor(self.v0, dtype=torch.float32)
                u0_t_batch = torch.tensor(self.u0_t, dtype=torch.float32)
                v0_t_batch = torch.tensor(self.v0_t, dtype=torch.float32)
    
                x_f_batch = torch.tensor(self.x_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
                t_f_batch = torch.tensor(self.t_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
    
                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, grads_u_lb, grads_u_ub, grads_col_v, grads_v, grads_v_lb, grads_v_ub = self.ggrad(self.model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, v0_batch, u0_t_batch, v0_t_batch, self.x_lb, self.t_lb, self.x_ub, self.t_ub)
    
                optimizer.zero_grad()
                for param, grad in zip(self.model.parameters(), grads):
                    param.grad = grad
                optimizer.step()
    
                # Apply gradients to col_weights and u_weights
                optimizer_col_weights.zero_grad()
                self.col_weights.grad = -grads_col
                optimizer_col_weights.step()
    
                optimizer_u_weights.zero_grad()
                self.u_weights.grad = -grads_u
                optimizer_u_weights.step()
    
                optimizer_u_lb_weights.zero_grad()
                self.u_lb_weights.grad = -grads_u_lb
                optimizer_u_lb_weights.step()
    
                optimizer_u_ub_weights.zero_grad()
                self.u_ub_weights.grad = -grads_u_ub
                optimizer_u_ub_weights.step()
    
                optimizer_col_v_weights.zero_grad()
                self.col_v_weights.grad = -grads_col_v
                optimizer_col_v_weights.step()
    
                optimizer_v_weights.zero_grad()
                self.v_weights.grad = -grads_v
                optimizer_v_weights.step()
    
                optimizer_v_lb_weights.zero_grad()
                self.v_lb_weights.grad = -grads_v_lb
                optimizer_v_lb_weights.step()
    
                optimizer_v_ub_weights.zero_grad()
                self.v_ub_weights.grad = -grads_v_ub
                optimizer_v_ub_weights.step()
    
            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error u,v: %.4e, %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
    
                start_time = time.time()

    def fit_lbfgs(self):
    
        batch_sz = self.N_f
        n_batches = self.N_f // batch_sz
    
        start_time = time.time()
        
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8, tolerance_grad=1e-05, tolerance_change=1e-09)
    
        print("starting L-BFGS training")
    
#        lossl_value= []
        for epoch in range(self.newton_iter):
            for i in range(n_batches):
    
                x0_batch = torch.tensor(self.x0, dtype=torch.float32)
                t0_batch = torch.tensor(self.t0, dtype=torch.float32)
                u0_batch = torch.tensor(self.u0, dtype=torch.float32)
                v0_batch = torch.tensor(self.v0, dtype=torch.float32)
                u0_t_batch = torch.tensor(self.u0_t, dtype=torch.float32)
                v0_t_batch = torch.tensor(self.v0_t, dtype=torch.float32)
    
                x_f_batch = torch.tensor(self.x_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
                t_f_batch = torch.tensor(self.t_f[i*batch_sz:(i*batch_sz + batch_sz),], dtype=torch.float32)
    
                loss_value, mse_0, mse_b, mse_f, grads, grads_col, grads_u, grads_u_lb, grads_u_ub, grads_col_v, grads_v, grads_v_lb, grads_v_ub = self.ggrad(self.model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, v0_batch, u0_t_batch, v0_t_batch, self.x_lb, self.t_lb, self.x_ub, self.t_ub)
    
                def closure():
                    optimizer.zero_grad()
                    for param, grad in zip(self.model.parameters(), grads):
                        param.grad = grad
    
                    return loss_value       
                
                optimizer.step(closure)

#            lossl_value.append(loss_value)#.item())    
#    
#            if epoch>10000:
#                a1= lossl_value[-1]
#                a2= lossl_value[-2]
#                a3= lossl_value[-3]
#                a4= lossl_value[-4]
#                if abs((a1-a2)/a2)<1e-10 and abs((a2-a3)/a3)<1e-10 and abs((a3-a4)/a4)<1e-10:
#                    break

            if (epoch+1) % 100 == 0:
                elapsed = time.time() - start_time
                error_u_value, error_v_value = self.error_u()
                print('It: %d, Time: %.2f, mse_0: %.4e, mse_f: %.4e, total loss: %.4e, Error u,v: %.4e, %.4e' % (epoch+1, elapsed, mse_0, mse_f, loss_value, error_u_value, error_v_value))
    
                start_time = time.time()

    def __Loadmat(self,fileName):

        data = scipy.io.loadmat(fileName)

        tt = data['t'].T[:,(self.id_t*tint):((self.id_t +1)*tint + 1)]
        t = tt.flatten()[:,None]

        x = data['x'].T.flatten()[:,None]
        self.Exact = data['Exact']#.T

        self.Exact_u = self.Exact.real.T
        X, T = np.meshgrid(x, t)
        self.x=x
        self.t=t
        self.X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        u_star_ori=self.Exact_u[:, (self.id_t*tint):((self.id_t +1)*tint + 1)]

        self.u_star=u_star_ori#.flatten()[:,None]

        # Domain bounds
        lb = self.X_star.min(0)
        ub = self.X_star.max(0)
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device, requires_grad=True)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device, requires_grad=True)

        idx_x = np.random.choice(x.shape[0], N0, replace=False)
        self.idx_x=idx_x
        x0 = x[idx_x,:]

#        self.u0 = tf.cast(self.u_star[idx_x,0:1], dtype = tfdoubstr)
        if self.id_t==0:
            self.u0 = torch.tensor(self.Exact_u[idx_x, 0:1], dtype=torch.float32).cuda()
        else:
            u0 = np.array(self.u0_t).flatten()[:,None]
            self.u0 = torch.tensor(u0[idx_x,0:1], dtype=torch.float32).cuda()
 
        idx_t = np.random.choice(t.shape[0], Nb, replace=False)
        self.idx_t=idx_t
        tb = t[idx_t,:]
        
        X_f = lb + (ub-lb)*lhs(2, self.N_f)
        self.x_f = torch.tensor(X_f[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_f = torch.tensor(X_f[:, 1:2]).float().requires_grad_(True).cuda()
        
        X0 = np.concatenate((x0, 0*x0 + t[0]), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        self.x0 = torch.tensor(X0[:, 0:1]).float().requires_grad_(True).cuda()
        self.t0 = torch.tensor(X0[:, 1:2]).float().requires_grad_(True).cuda()
        
        self.x_lb = torch.tensor(X_lb[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_lb = torch.tensor(X_lb[:, 1:2]).float().requires_grad_(True).cuda()
        self.x_ub = torch.tensor(X_ub[:, 0:1]).float().requires_grad_(True).cuda()
        self.t_ub = torch.tensor(X_ub[:, 1:2]).float().requires_grad_(True).cuda()

        u_lb_all = self.u_star[0,  :].flatten()[:,None]
        u_ub_all = self.u_star[-1, :].flatten()[:,None]
        u_lb= u_lb_all[idx_t]
        u_ub= u_ub_all[idx_t]
        self.u_lb = torch.tensor(u_lb).float().requires_grad_(True).cuda()
        self.u_ub = torch.tensor(u_ub).float().requires_grad_(True).cuda()

        self.Exact_v=self.Exact.imag.T
        v_star_ori=self.Exact_v[:, (self.id_t*tint):((self.id_t +1)*tint + 1)]
        self.v_star=v_star_ori
        if self.id_t==0:
            self.v0 = torch.tensor(self.Exact_v[self.idx_x, 0:1], dtype=torch.float32).cuda()
        else:
            v0 = np.array(self.v0_t).flatten()[:,None]
            self.v0 = torch.tensor(v0[self.idx_x,0:1], dtype=torch.float32).cuda()
        v_lb_all = self.v_star[0,  :].flatten()[:,None]
        v_ub_all = self.v_star[-1, :].flatten()[:,None]
        v_lb= v_lb_all[self.idx_t]
        v_ub= v_ub_all[self.idx_t]
        self.v_lb = torch.tensor(v_lb).float().requires_grad_(True).cuda()
        self.v_ub = torch.tensor(v_ub).float().requires_grad_(True).cuda()

    def error_u(self):
#        u_pred,v_pred, f_u_pred, f_v_pred = self.predict()#self.u_model, self.X_star)
        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_pred,v_pred,_, _ = self.uv_model(X_star[:, 0:1], X_star[:, 1:2])
        u_star = self.Exact_u.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
        u_star = u_star.flatten()[:, None]
        
        v_star = self.Exact_v.T[(self.id_t*tint):((self.id_t +1)*tint + 1),:]
        v_star = v_star.flatten()[:, None]

        u_pred= u_pred.detach().cpu().numpy()
        v_pred= v_pred.detach().cpu().numpy()
        
        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        
        return error_u, error_v

    def predict(self):
        X_star = torch.tensor(self.X_star, dtype=torch.float32, device=device, requires_grad=True)
        u_pred,v_pred,_, _ = self.uv_model(X_star[:, 0:1], X_star[:, 1:2])

        X_star = X_star.clone().detach().requires_grad_(True)
        f_u_pred,f_v_pred = self.f_model(self, X_star[:, 0:1], X_star[:, 1:2])

        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()
        f_u_pred = f_u_pred.detach().cpu().numpy()
        f_v_pred = f_v_pred.detach().cpu().numpy()

        return u_pred,v_pred, f_u_pred,f_v_pred
