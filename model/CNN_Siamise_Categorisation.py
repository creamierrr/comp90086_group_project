from model.__template__ import *
from model.Modules import *

class CNN_Siamise_Categorisation(CNN_Categorisation_Model): # ALL COMMENTS SAME AS CNN_SIAMESE_TRIPLET UNLESS ALTERNATIVELY COMMENTED
    class Model(nn.Module):
                
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            assert not (self.CFG.freeze_encoder and not self.CFG.pretrained), "If encoder is frozen, it must be pretrained"

            torch.manual_seed(self.CFG.random_state)

            self.encoder = torch.hub.load('pytorch/vision:v0.10.0', self.CFG.encoder, pretrained=self.CFG.pretrained) if type(self.CFG.encoder) == str else self.CFG.encoder

            if self.CFG.crop_pretrained_linear:
                self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])


            sample_input = torch.randn([1, self.CFG.input_shape[1], self.CFG.input_shape[2], self.CFG.input_shape[3]])  
            sample_output = self.encoder(sample_input)

            flatten_shape = np.prod(sample_output.shape[1:])

            for param in self.encoder.parameters():
                param.requires_grad = not self.CFG.freeze_encoder

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

            if self.CFG.num_mlp_layers:
                
                self.transition = nn.Linear(flatten_shape, self.CFG.hidden_dim)
                self.transition = nn.Linear(flatten_shape, self.CFG.hidden_dim)

                if self.CFG.res_learning:
                    self.mlp = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)]) # -1 layer because of transition layer
                    self.mlp = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                else:
                    self.mlp = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                    self.mlp = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])

                self.out = nn.Linear(self.CFG.hidden_dim*2, 2)

            else:

                self.out = nn.Linear(flatten_shape*2, 2)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.CFG.dropout)
        
            
        def forward(self, x_left, x_right):
            
            x_left = x_left.permute(0, 3, 1, 2)
            x_right = x_right.permute(0, 3, 1, 2)

            x_left = self.encoder(x_left)
            x_right = self.encoder(x_right)

            batch_size = x_left.size(0)

            x_left = x_left.reshape(batch_size, -1)
            x_right = x_right.reshape(batch_size, -1)

            if self.CFG.num_mlp_layers:
                x_left = self.dropout(self.relu(self.transition(x_left)))
                x_right = self.dropout(self.relu(self.transition(x_right)))

                for layer in self.mlp:
                    x_left = layer(x_left)
                    x_right = layer(x_right)

            # concat and feed to final classification layer 
            x_combined_embed = torch.cat((x_left, x_right), 1)

            return self.softmax(self.out(x_combined_embed))
        
    def __init__(self, CFG, name="CNN_Siamise_Categorisation"):
        super().__init__(CFG, name=CFG.name)