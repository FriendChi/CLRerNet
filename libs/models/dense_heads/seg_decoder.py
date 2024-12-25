import torch.nn.functional as F
from torch import nn

class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
       
        #self.norm = LayerNorm(n_feats, data_format='channels_first')
        #self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())
        
        self.att = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats),  
                nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9//2)*3, groups=n_feats, dilation=3),
                nn.Conv2d(n_feats, n_feats, 1, 1, 0))  

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv0(x)
        x = x*self.att(x) 
        x = self.conv1(x) 
        return x    

class SegDecoder(nn.Module):
    """
    Segmentation decoder head for auxiliary loss.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/seg_decoder.py
    """

    def __init__(
        self,
        image_height,
        image_width,
        num_classes,
        prior_feat_channels=64,
        refine_layers=3,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * refine_layers, num_classes, 1)
        self.image_height = image_height
        self.image_width = image_width
        self.mlka = LKAT(prior_feat_channels * refine_layers)

    def forward(self, x):
        x = self.dropout(x)
        x = self.mlka(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=[self.image_height, self.image_width],
            mode="bilinear",
            align_corners=False,
        )
        return x
