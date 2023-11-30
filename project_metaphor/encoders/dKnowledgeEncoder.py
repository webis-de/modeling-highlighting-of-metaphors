import sys
sys.path.append('/Users/msen/Documents/DFG-trr318-explainability-20-today/c04-metaphors-data/Highlight/preprocessing')
import torch 
from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from parse_dataset_csr import csr_only
print('pass')
# Load the DistillBERT tokenizer.
print('Loading DistillBERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

################### Tokenize dataset ################################
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []
# For every sentence...
for each in csr_only:                                           # insert list of data to crunch
    encoded_dict = tokenizer.encode_plus(
                        each,                                   # Sentence to encode.
                        add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                        max_length = 64,                        # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,           # Construct attn. masks.
                        return_tensors = 'pt',                  # Return pytorch tensors.
                   )
      
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',                                                            
        output_hidden_states = True,                           
        return_dict=True
    )    

res = model(input_ids, attention_mask=attention_masks)
h_sts = res['hidden_states'][6] # take the last layer

class ScaledDotProductAttention(nn.Module):
    # Scaled Dot-Product Attention
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

'''
# d_word_vec=512, d_model=512, d_inner=2048,
# n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200
'''

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, n_head=4, d_k=128, d_v=128, dropout=0.1):
        super(Multi_Head_Attention,self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        sz_b, len_q, len_k, len_v = 128, 1, 1, 1

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
    
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual

        q = self.layer_norm(q)
        q = q.reshape(1,sz_b,d_model)

        return q, attn

class Norm_layer(nn.Module):
    def __init__(self):
        super(Norm_layer, self).__init__()
        self.layer_norm = nn.LayerNorm([64,768], eps=1e-6)

    def forward(self,x):
        return self.layer_norm(x)
    
class Feed_forward(nn.Module):
    def __init__(self):
        super(Feed_forward, self).__init__()
        self.layer_norm = nn.Linear([64,768], eps=1e-6)

    def forward(self,x):
        return self.layer_norm(x)

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ 
            Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        # print(nf,nx)
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Norm_layer_2(nn.Module):
    def __init__(self):
        super(Norm_layer_2, self).__init__()
        self.layer_norm = nn.LayerNorm([128, 128], eps=1e-6)

    def forward(self,x):
        return self.layer_norm(x)

d_model = 768
def execute():
    out_sts=list()
    for i in range(len(res['hidden_states'][6])):
        h_sts = res['hidden_states'][6][i]
        # is this even OK to do? 
        q = h_sts
        k = h_sts
        v = h_sts
        attn = Multi_Head_Attention(d_model)
        norm = Norm_layer()
        conv1D = Conv1D(128,768)
        output, attn1 = attn(q,k,v)
        output = norm(output[0])
        output = conv1D(output)
        norm2 = Norm_layer_2()
        output = norm2(output)
        out_sts.append(output)

    return out_sts 