import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fbsde(): #class to initialize our equation
    def __init__(self, x_0, b, sigma, f, g, T, dim_x,dim_y,dim_d):
        self.x_0 = x_0.to(device)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d


class Model_EM(nn.Module):
    def __init__(self, equation, dim_h, num_h):
        super(Model_EM, self).__init__()
        self.linear_input = nn.Linear(equation.dim_x+1, dim_h) #input layer

        #hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_h - 1):
            self.hidden_layers.append(nn.Linear(dim_h, dim_h))

        #batch normalization layers
        self.batch_norm = nn.BatchNorm1d(dim_h)
        self.linear_output = nn.Linear(dim_h, equation.dim_y*equation.dim_d) #output layer

        #initial parameters
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device)) #initial y_0 for CIR model
        #self.y_0 = nn.Parameter(55 + 10 * torch.rand(equation.dim_y, device=device)) #initial y_0 for black scholes PDE
        self.equation= equation


    def forward(self,batch_size, N):
        def phi(x): #network that we are learning, = z in our backwards equation
            x = F.relu((self.linear_input(x)))
            for layer in self.hidden_layers:
                x = F.relu((layer(x)))
            return self.linear_output(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)

        delta_t = self.equation.T / N #time step
        
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t) #brownian motion
       
        x = self.equation.x_0+torch.zeros(W.size()[0],self.equation.dim_x,device=device) #initialise forwards equation
        y = self.y_0+torch.zeros(W.size()[0],self.equation.dim_y,device=device) #initialise backwards equation

        for i in range(N):
            u = torch.cat((x, torch.ones(x.size()[0], 1,device=device)*delta_t*i), 1)
            z = phi(u) #approx z from the network
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)

            x = x+self.equation.b(delta_t*i, x, y)*delta_t+torch.matmul( self.equation.sigma(delta_t*i, x), w).reshape(-1, self.equation.dim_x) #simulate forwards equation
            y = y - self.equation.f(delta_t*i, x, y, z)*delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y) #simulate backwards equation
        return x, y


class BSDEsolverEm():
    def __init__(self, equation, dim_h, num_h):
        self.model = Model_EM(equation,dim_h, num_h).to(device)
        self.equation = equation

    def train(self, batch_size, N, itr, log, test_num,):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters()) #optimizer

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
            np.save(f'loss_data_EM{test_num}', loss_data)
            np.save(f'y0_data_EM{test_num}', y0_data)
