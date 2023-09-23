from model.__template__ import *
from model.Modules import *

class CNN_Key_Query_Triplet(CNN_Triplet_Model):
    class Model(nn.Module):
                
        def __init__(self, CFG):
            super().__init__()
            self.CFG = CFG

            assert not (self.CFG.freeze_encoder and not self.CFG.pretrained), "If encoder is frozen, it must be pretrained"

            torch.manual_seed(self.CFG.random_state)

            self.encoder_query = torch.hub.load('pytorch/vision:v0.10.0', self.CFG.encoder, pretrained=self.CFG.pretrained)
            self.encoder_key = torch.hub.load('pytorch/vision:v0.10.0', self.CFG.encoder, pretrained=self.CFG.pretrained)

            sample_input = torch.randn(self.CFG.input_shape)  
            sample_output = self.encoder_query(sample_input)

            flatten_shape = np.prod(sample_output.shape[1:])

            for param in self.encoder_query.parameters():
                param.requires_grad = not self.CFG.freeze_encoder
            
            for param in self.encoder_key.parameters():
                param.requires_grad = not self.CFG.freeze_encoder

            if self.CFG.num_mlp_layers:
                
                self.transition_query = nn.Linear(flatten_shape, self.CFG.hidden_dim)
                self.transition_key = nn.Linear(flatten_shape, self.CFG.hidden_dim)

                if self.CFG.res_learning:
                    self.mlp_query = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)]) # -1 layer because of transition layer
                    self.mlp_key = nn.ModuleList([ResLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                else:
                    self.mlp_query = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])
                    self.mlp_key = nn.ModuleList([LinearLayer(self.CFG) for _ in range(self.CFG.num_mlp_layers-1)])

                self.out_query = nn.Linear(self.CFG.hidden_dim, self.CFG.embed_dim)
                self.out_key = nn.Linear(self.CFG.hidden_dim, self.CFG.embed_dim)

            else:
                self.out_query = nn.Linear(flatten_shape, self.CFG.embed_dim)
                self.out_key = nn.Linear(flatten_shape, self.CFG.embed_dim)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(self.CFG.dropout)
        
            
        def forward(self, x_anchor, x_positive, x_negative):
            
            x_anchor = x_anchor.permute(0, 3, 1, 2) # change axis order
            x_positive = x_positive.permute(0, 3, 1, 2)
            x_negative = x_negative.permute(0, 3, 1, 2)

            x_anchor = self.encoder_query(x_anchor)
            x_positive = self.encoder_key(x_positive)
            x_negative = self.encoder_key(x_negative)

            batch_size = x_anchor.size(0)
            
            x_anchor = x_anchor.reshape(batch_size, -1) # flatten out the encoded image
            x_positive = x_positive.reshape(batch_size, -1)
            x_negative = x_negative.reshape(batch_size, -1)

            if self.CFG.num_mlp_layers:
                x_anchor = self.dropout(self.relu(self.transition_query(x_anchor))) # first go through resizing layer to change flatten_size to hidden dim
                x_positive = self.dropout(self.relu(self.transition_key(x_positive)))
                x_negative = self.dropout(self.relu(self.transition_key(x_negative)))

                for layer in self.mlp_query: # go through mlp layers
                    x_anchor = layer(x_anchor)
                for layer in self.mlp_key:
                    x_positive = layer(x_positive)
                    x_negative = layer(x_negative)
    
            return self.relu(self.out_query(x_anchor)), self.relu(self.out_key(x_positive)), self.relu(self.out_key(x_negative)) # export embedding
        
    def __init__(self, CFG, name="CNN_Triplet"):
        super().__init__(CFG, name=name)