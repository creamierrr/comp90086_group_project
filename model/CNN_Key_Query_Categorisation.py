from model.__template__ import *
from model.Modules import *

class CNN_Key_Query_Categorisation(CNN_Categorisation_Model):
    class Model(nn.Module):
                
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            assert not (self.CFG.freeze_encoder and not self.CFG.pretrained), "If encoder is frozen, it must be pretrained"

            torch.manual_seed(self.CFG.random_state)

            self.encoder_left = torch.hub.load('pytorch/vision:v0.10.0', self.CFG.encoder, pretrained=self.CFG.pretrained)
            self.encoder_right = torch.hub.load('pytorch/vision:v0.10.0', self.CFG.encoder, pretrained=self.CFG.pretrained)

            self.encoder_left = nn.Sequential(*list(self.encoder_left.children())[:-1])
            self.encoder_right = nn.Sequential(*list(self.encoder_right.children())[:-1])

            if self.CFG.crop_pretrained_linear:
                sample_input = torch.randn(self.CFG.input_shape)  
                sample_output = self.encoder_left(sample_input)

            flatten_shape = np.prod(sample_output.shape[1:])

            for param in self.encoder_left.parameters():
                param.requires_grad = not self.CFG.freeze_encoder
            
            for param in self.encoder_right.parameters():
                param.requires_grad = not self.CFG.freeze_encoder

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

            if self.CFG.num_mlp_layers:
                
                self.transition_left = nn.Linear(flatten_shape, self.CFG.hidden_dim)
                self.transition_right = nn.Linear(flatten_shape, self.CFG.hidden_dim)

                if self.CFG.res_learning:
                    self.mlp_left = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)]) # -1 layer because of transition layer
                    self.mlp_right = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                else:
                    self.mlp_left = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                    self.mlp_right = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])

                self.out = nn.Linear(self.CFG.hidden_dim*2, 2)

            else:

                self.out = nn.Linear(flatten_shape*2, 2)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.CFG.dropout)
        
            
        def forward(self, x_left, x_right):
            
            x_left = x_left.permute(0, 3, 1, 2)
            x_right = x_right.permute(0, 3, 1, 2)

            x_left = self.encoder_left(x_left)
            x_right = self.encoder_right(x_right)

            batch_size = x_left.size(0)

            x_left = x_left.reshape(batch_size, -1)
            x_right = x_right.reshape(batch_size, -1)

            if self.CFG.num_mlp_layers:
                x_left = self.dropout(self.relu(self.transition_left(x_left)))
                x_right = self.dropout(self.relu(self.transition_right(x_right)))

                for layer in self.mlp_left:
                    x_left = layer(x_left)
                for layer in self.mlp_right:
                    x_right = layer(x_right)
                    x_right = layer(x_right)
                
            x_combined_embed = torch.cat((x_left, x_right), 1)

            return self.softmax(self.out(x_combined_embed))
        
    def __init__(self, CFG, name="CNN_Categorisation"):
        super().__init__(CFG, name=CFG.name)