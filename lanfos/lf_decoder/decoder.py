from torch import nn
import torch

class ResidualLinear(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        x = x + self.linear(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dim_list, cluster_centers=None, torch_device="cuda"):
        super().__init__()
        if cluster_centers is not None:
            self.cluster_centers = cluster_centers.to(torch_device)
            self.cluster_centers.requires_grad_(False)

        self.mlp = nn.ModuleList()
        for i in range(len(dim_list)-1):
            if dim_list[i]==dim_list[i+1]:
                self.mlp.append(ResidualLinear(dim_list[i], bias=True))
            else:
                self.mlp.append(nn.Linear(dim_list[i], dim_list[i+1], bias=True))

            if i<len(dim_list)-2:
                self.mlp.append(nn.LeakyReLU())
        
         # input to this layer has shape [..., N] so we do softmax over the "N" dimension
        self.softmax = nn.Softmax(dim=-1)
        
    def get_distribution(self, x):
        for layer in self.mlp:
            x = layer(x)
        x = self.softmax(x)
        return x
    
    def forward(self, x):
        x = self.get_distribution(x)
        # x has shape [..., N] and cluster_centers has shape [N, D]
        # Finally the x which is returned has shape [..., D]
        x = torch.matmul(x, self.cluster_centers)
        return x
