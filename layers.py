import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Union, Callable, Optional
import copy

from utility.preprocessing import *
from transformers import AutoTokenizer, BertConfig, AutoModel

class CLP_clinical(nn.Module):
    def __init__(self,
                bert_model_name: str,
                embed_dim: int = 768,
                freeze_layers:Union[Tuple[int, int], int] = None):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):#12
        try:
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)  #bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)#, return_dict=True)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(input_ids = text['input_ids'].cuda(), attention_mask = text['attention_mask'].cuda())
        last_hidden_state, pooler_output, hidden_states = output[0],output[1],output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out
    
    def forward(self,text1,text2):
        text1_features = self.encode_text(text1)
        text2_features = self.encode_text(text2)
        text1_features = F.normalize(text1_features, dim=-1)
        text2_features = F.normalize(text2_features, dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()
    
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, text_features, image_features):
        output = text_features
        for n,layer in enumerate(self.layers):
            output= layer(output, image_features)
        output = self.norm(output)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,batch_first=True)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        #self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, text_features, image_features):

        text_features1,attn_weights1 = self.self_attn(text_features, text_features, text_features)
        text_features = self.norm1(text_features + self.dropout1(text_features1))

        features,attn_weights2 = self.multihead_attn(text_features,image_features,image_features)
        features = self.norm2(text_features + self.dropout2(features))

        features1 = self.linear2(self.dropout(F.relu(self.linear1(features))))
        features = features + self.dropout3(features1)

        return features
    
class co_GraphConvolution(nn.Module):

    def __init__(self, bias=True):
        super(co_GraphConvolution, self).__init__()
        self.bias = bias
        self.linear = nn.Linear(768, 768, bias)
        
    def forward(self, input):
        support = self.linear(input)
        coM = np.loadtxt('CheXpert14_0_coM.txt')                #(14,14)
        coM = torch.from_numpy(coM).cuda().to(torch.float32)   #torch.Size([14, 14])
        output = torch.matmul(coM, support)                    #[14,768]
        return output

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, norm='', bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(512, 512, bias)
        self.norm=norm
        
    def forward(self, input, adj=1.0):
        #print(adj)
        input = to_dense(input) 
        support = self.linear(input)
        support = support.view(support.shape[0],-1)
        #support = input
        if isinstance(adj, (float, int)):
            output = support*adj  
        else:
            adj = adj_norm(adj, True) if self.norm=='symmetric' else adj_norm(adj, False) if self.norm=='asymmetric' else adj  #矩阵[8,40]
            output = torch.matmul(adj, support)  #[8,14]
        output = output.view(output.shape[0],49,512)
        return output

    def __repr__(self):
        return self.__class__.__name__ +'(in_features={}, out_features={}, bias={}, norm={})'.format(
            self.in_features, self.out_features, self.bias, self.norm )
    

    
class MyVggNet16_bn(nn.Module):
    def __init__(self, outnum=14, gpsize=4,inchannel=3):
        super(MyVggNet16_bn, self).__init__()
        original_model = models.vgg16_bn(pretrained=True)
        if inchannel!=3:
            original_model.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            
        self.features = original_model.features #torch.Size([batch-size, 512, 7, 7])

        self.classifier = nn.Linear(1024, outnum)
        
    def forward(self, x):
        x = self.features(x)       
        #x = x.view(-1,1024)
        x = self.classifier(x)
        return x
   
    