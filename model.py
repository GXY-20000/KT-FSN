import scipy.sparse as sp
import copy
import torch
from torch import nn
import torch.nn.functional as F

from layers import *

from utility.preprocessing import str2value, issymmetric, to_dense, to_sparse


class MRGCN(nn.Module):
    """
    multi-relational GCN
    """
    def __init__(self, relations, in_dim=None, out_dim=None, enc='vgg16bn', inchannel=3, selfweight=1):
        super(MRGCN, self).__init__()
        self.selfweight = selfweight
        self.encoder = MyVggNet16_bn(outnum=out_dim,inchannel=inchannel).features

        self.gcn = nn.ModuleDict({str(i): GraphConvolution(in_dim, out_dim) for i in relations+['self']})
        self.co_gcn = co_GraphConvolution()

        self.image_linear = nn.Linear(512, 768)

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_encoder = CLP_clinical(bert_model_name="emilyalsentzer/Bio_ClinicalBERT")

        self.norm = nn.LayerNorm(normalized_shape=768)
        self.TransformerDecoderLayer = TransformerDecoderLayer(embed_dim=768,num_heads=4)
        self.TransformerDecoder = TransformerDecoder(self.TransformerDecoderLayer,num_layers=4,norm=self.norm)
        self.classifier = nn.Linear(768, 1)
    
    def forward(self, fea_in, k, adj_mats):                                                               #[batch-size*5,channel:3,224,224] 
        gcn_features_all = self.encoder(fea_in)                                                           #torch.Size([batch-size+batch-size*5, 512, 7, 7])                                          
        gcn_features_all = gcn_features_all.view(gcn_features_all.shape[0],gcn_features_all.shape[1],-1)  #torch.Size([batch-size+batch-size*5, 512,7*7])
        gcn_features_all = torch.transpose(gcn_features_all, 1, 2)                                        #torch.Size([batch-size, 49, 512]) 
        gcn_features_self = gcn_features_all[k:]                                                          #torch.Size([batch-size, 512, 49])
        gcn_features = self.gcn['self'](gcn_features_self, self.selfweight)                               #torch.Size([batch-size, 512, 49])

        for i, adj in adj_mats.items():
            gcn_features = gcn_features + self.gcn[i](gcn_features_all, adj)            #torch.Size([batch-size, 512, 49])
        
        image_features = torch.transpose(image_features, 1, 2)                          #torch.Size([batch-size, 49, 512])
        image_features = self.image_linear(gcn_features)                                #torch.Size([batch-size, 49, 768])

        text_list = ['NoFinding','Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion','Edema', 'Consolidation', \
                    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture' , 'Support Devices']#
        '''text_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', \
                     'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']'''
        
        text_token =  self.tokenizer(list(text_list),add_special_tokens=True,padding=True,return_tensors='pt')
        text_features = self.text_encoder.encode_text(text_token)                   #torch.Size([out_dim, 768])
        text_features = self.co_gcn(text_features)                                  #torch.Size([out_dim, 768])
        text_features = text_features.unsqueeze(0)                                  #torch.Size([1, out_dim, 768])
        text_features = text_features.repeat(image_features.shape[0], 1, 1)         #torch.Size([batch-size, out_dim, 768])

        image_features = self.norm(image_features)
        text_features = self.norm(text_features)                                                      
        fea_out = self.TransformerDecoder(text_features, image_features)                                    #torch.Size([batch-size, out_dim, 768])
        fea_out = self.classifier(fea_out)                                                                  #torch.Size([batch-size, out_dim, 1])
        fea_out = fea_out.squeeze(2)                                                                        #torch.Size([batch-size, out_dim])

        '''import pdb
        pdb.set_trace()'''

        return fea_out

class ImageGCN(nn.Module):
    def __init__(self, hid_dims, out_dims, relations, encoder='alex', inchannel=3, share_encoder=False, dropout=0.1):
        super(ImageGCN, self).__init__()
        self.imagelayer = MRGCN(relations, enc=encoder, inchannel=inchannel)
        self.denselayer = MRGCN(relations, hid_dims, out_dims)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    
    def forward(self, fea, adj_mats2, adj_mats1, k=8 ):
        fea2 = self.imagelayer(fea, fea, adj_mats1)
        fea2 = fea2.view(-1, 1024)
        fea2 = self.relu(self.dropout(fea2))
        fea = self.denselayer(fea2, fea2[k:], adj_mats2)
        return fea
    
class SingleLayerImageGCN(nn.Module):
    def __init__(self, relations, encoder='singlevgg16bn', in_dim=1024,out_dim=14, inchannel=3):
        super(SingleLayerImageGCN, self).__init__()
        self.out_dim = out_dim
        self.layer = MRGCN(relations, enc=encoder,in_dim=in_dim,out_dim=out_dim, inchannel=inchannel)        
    
    def forward(self, fea, adj_mats, k):              
        fea = self.layer(fea, k, adj_mats)                            #[batch-size,疾病数：14]                                            
        fea = fea.view(-1, self.out_dim)                             #[batch-size,疾病数：14]
        #print(fea2.shape)
        return fea
    
class W_BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(W_BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        pos = torch.abs((target!=0).sum())
        neg = torch.abs((target==0).sum())
        pos_weight = (pos+neg+1)/(pos+1)
        neg_weight = (pos+neg+1)/(neg+1)
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=pos_weight,weight=neg_weight)

class W_BCELossWithNA(nn.Module):
    def __init__(self, reduction='mean'):
        super(W_BCELossWithNA, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):  
        pos = torch.abs((target!=0).sum())
        neg = torch.abs((target==0).sum())
        pos_weight = (pos+neg+1)/(pos+1)
        neg_weight = (pos+neg+1)/(neg+1)  
        return F.binary_cross_entropy_with_logits(input[target!=-1], target[target!=-1], reduction=self.reduction, pos_weight=pos_weight,weight=neg_weight)   

