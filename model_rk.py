import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model_RK(nn.Module):
    def __init__(self, equation, dim_h, num_h):
        super(Model_RK, self).__init__()
        self.linear_input = nn.Linear(equation.dim_x+1, dim_h) #input layer
        #hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_h - 1):
            self.hidden_layers.append(nn.Linear(dim_h, dim_h))
        self.linear_output = nn.Linear(dim_h, equation.dim_y*equation.dim_d) #output layer
        
        #initial parameters
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device)) #initial y_0 for CIR model
        #self.y_0 = nn.Parameter(55 + 10 * torch.rand(equation.dim_y, device=device)) #initial y_0 for black scholes PDE

    def forward(self,batch_size, N):
        def phi(x): #network that we are learning, = z in our backwards equation
            x = F.relu((self.linear_input(x)))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            return self.linear_output(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)


        delta_t = self.equation.T / N #time step
        
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t) #brownian motion
       
        x = self.equation.x_0+torch.zeros(W.size()[0],self.equation.dim_x,device=device) #initialise forwards equation
        y = self.y_0+torch.zeros(W.size()[0],self.equation.dim_y,device=device) #initialise backwards equation

        for i in range(N):
            u = torch.cat((x, torch.ones(x.size()[0], 1,device=device)*delta_t*i), 1)
            z = phi(u) #approx z from the network
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)

            #terms for Runge-Kutta (forwards)
            S = np.random.choice([-1, 1])
            k1 = self.equation.b(delta_t*i, x, y)*delta_t + torch.matmul(self.equation.sigma(delta_t*i, x), (w - S*np.sqrt(delta_t))).reshape(-1, self.equation.dim_x)
            k2 =self.equation.b(delta_t*(i+1), x+k1, y)*delta_t + torch.matmul( self.equation.sigma(delta_t*(i+1), x+k1), w + S*np.sqrt(delta_t)).reshape(-1, self.equation.dim_x)
            
            #terms for Runge-Kutta (backwards)
            y1 = -self.equation.f(delta_t*i, x, y, z)*delta_t + torch.matmul(z, w - S*np.sqrt(delta_t)).reshape(-1, self.equation.dim_y)
            z_new = phi(torch.cat((x+k1, torch.ones(x.size()[0], 1,device=device)*delta_t*(i+1)), 1)) #approx z from the network
            y2 = -self.equation.f(delta_t*(i+1), x, y+ y1, z)*delta_t + torch.matmul(z_new, w + S*np.sqrt(delta_t)).reshape(-1, self.equation.dim_y)
            
            
            x = x + (k1 + k2)/2 #simulate forwards equation
            y = y + (y1 + y2)/2 #simulate backwards equation
        return x, y


class BSDEsolverRK():
    def __init__(self, equation, dim_h, num_h):
        self.model = Model_RK(equation,dim_h, num_h).to(device)
        self.equation = equation

    def train(self, batch_size, N, itr, log, test_num):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters())

        loss_data, y0_data = [], []

        for i in range(itr):
            x, y = self.model(batch_size,N)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if log:
            np.save(f'loss_data_RK_fwd{test_num}', loss_data)
            np.save(f'y0_data_RK_fwd{test_num}', y0_data)
