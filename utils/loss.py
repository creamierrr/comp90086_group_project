from environment import *

class Triangular_Triplet_Loss(nn.Module):
    def __init__(self, margin=1.0, scale = 1):
        super().__init__()
        self.margin = margin
        self.scale = scale
    def forward(self, anchor, positive, negative):
        
        anc_pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        
        anc_neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        pos_neg_dist = torch.sum((positive - negative) **2, dim = 1)
        
        loss = torch.relu(self.scale * anc_pos_dist - anc_neg_dist - pos_neg_dist + self.margin)
        return loss.mean()


class Double_Triplet_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        
        anc_pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        
        anc_neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

        pos_neg_dist = torch.sum((positive - negative) **2, dim = 1)
        
        loss = torch.relu(anc_pos_dist - anc_neg_dist+self.margin) + torch.relu(anc_pos_dist - pos_neg_dist+self.margin)
        return loss.mean()