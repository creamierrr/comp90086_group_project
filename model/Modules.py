from environment import *

class LinearLayer(nn.Module):
    """ Dense Layer """
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG.random_state)

        self.layer = nn.Sequential(
            nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.CFG.dropout)
        )
        
    def forward(self, x):
        return self.layer(x)
    
    
class ResLayer(nn.Module):
    """ Dense Residual Layer """
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        torch.manual_seed(self.CFG.random_state)

        self.layer = nn.Sequential(
            nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.CFG.hidden_dim, self.CFG.hidden_dim)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.CFG.dropout)
        
    def forward(self, x):
        return self.dropout(self.activation(x + self.layer(x)))