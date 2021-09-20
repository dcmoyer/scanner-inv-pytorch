

import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder( nn.Module ):
    
    def __init__(self, n_in, n_out):
        super(encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append( nn.Linear( n_in, 256 ) )
        self.layers.append( nn.Linear( 256, 128 ) )
        self.layers.append( nn.Linear( 128, 64 ) )

        self.mu_head = nn.Linear( 64, n_out )
        self.log_sigma_sq_head = nn.Linear( 64, n_out )

    def forward( self, x ):
        x = torch.tanh( self.layers[0](x) )
        x = torch.tanh( self.layers[1](x) )
        x = torch.tanh( self.layers[2](x) )
        return torch.tanh( self.mu_head(x) ), self.log_sigma_sq_head(x)


class decoder( nn.Module ):

    def __init__(self, n_in, n_out, n_c):
        super(decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append( nn.Linear( n_in + n_c , 64 ) )
        self.layers.append( nn.Linear( 64, 128 ) )
        self.layers.append( nn.Linear( 128, 256 ) )
        self.layers.append( nn.Linear( 256, n_out ) )

    def forward( self, x, c):
        x = torch.cat((x,c),dim=1)
        x = torch.tanh( self.layers[0](x) )
        x = torch.tanh( self.layers[1](x) )
        x = torch.tanh( self.layers[2](x) )
        return self.layers[3](x)


class adv( nn.Module ):

    def __init__(self, n_in, n_out ):
        super(adv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append( nn.Linear( n_in, 32 ) )
        self.layers.append( nn.Linear( 32, 32 ) )
        self.layers.append( nn.Linear( 32, n_out ) )

    def forward(self, x):
        x = torch.tanh( self.layers[0](x) )
        x = torch.tanh( self.layers[1](x) )
        return torch.sigmoid( self.layers[2](x) )





