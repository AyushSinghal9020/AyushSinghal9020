import numpy
import numpy as np 
import pandas as pd 
import warnings
import scipy

def convertor(val , return_val = None):
    
    if return_val : 
        if isinstance(val , list): val = np.array(val)
    else : 
        if isinstance(val , numpy.ndarray) : val = val.tolist()
    
    return val

def matmul(f_row , f_col , s_row , s_col):
    
    if len(f_col) != len(s_row) : raise ValueError(f'Cannot multiply array with dimensions {f_row}x{f_col} , {s_row}x{s_col}')
    
    f_row = convertor(f_row , return_val = True)
    f_col = convertor(f_col , return_val = True)
    s_row = convertor(s_row , return_val = True)
    s_col = convertor(s_col , return_val = True)
    
    val = sum(f_col * s_row / 2)
    col = s_col * val
    
    return col

def layer_norm(val):
    
    val = (val - val.min()) / (val.max() - val.min())
    
    return val

def padding(val , padding_length):
    
    val = np.concatenate([
        val , 
        np.zeros(shape = padding_length - len(val) , 
                 dtype = np.int8)
    ])
    
    return val

class linear: 

    def __init__(self , in_feats , out_feats):

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.in_col = np.random.randn(self.in_feats)
        self.out_col = np.random.randn(self.out_feats)

    def forward(self , inps):

        if len(inps.shape) == 1 : 
            return_val = matmul(self.in_col , inps , 
                                self.in_col , self.out_col)
        
        elif len(inps.shape) == 2 : return_val = np.vstack([
            matmul(self.in_col , val , self.in_col , self.out_col)
            for val in inps])

        elif len(inps.shape) == 3 :
            return_val = []

            for batch in inps:

                batched = np.vstack([
                    matmul(self.in_col , val , self.in_col , self.out_col)
                    for val in batch])
            
                return_val.append(batched)

            return_val = np.stack(return_val)

        else : raise ValueError(f'Inputs of shape {inps.shape}cannot be sent to processed')

        return return_val
    
class embedding: 

    def __init__(self , in_feats , out_feats): 
        
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.feats = np.random.rand(self.in_feats , self.out_feats)
        self.parameters = [self.feats]

    def forward(self , inps):

        if len(inps.shape) == 1 : return_val = np.vstack([
            self.feats[val] for val in inps])
        
        elif len(inps.shape) == 2 : return_val = np.stack(
            [np.vstack([
                self.feats[value] for value in val
            ]) for val in inps])

        elif len(inps.shape) == 3 : 

            return_val = []

            for batch in inps:

                batched = [[self.feats[value] for value in val]
                           for val in batch]
                
                return_val.append(batched)

            return_val = np.stack(return_val)

        else : raise ValueError(f'Cannot process inputs with shape{inps.shape}')

        return return_val
    
class mhsa :

    def __init__(self , vocab_size):

        self.vocab_size = vocab_size

        self.queries = linear(self.vocab_size , self.vocab_size)
        self.keys = linear(self.vocab_size , self.vocab_size)
        self.values = linear(self.vocab_size , self.vocab_size)

        self.parameters = [[self.queries.in_col , self.queries.out_col] , 
                           [self.keys.in_col , self.keys.out_col] , 
                           [self.values.in_col , self.values.out_col]]
        
    def forward(self , query , key , value , mask = None):

        query_output = self.queries.forward(query)
        key_output = self.keys.forward(key)
        value_output = self.values.forward(value)

        attention = (query_output * key_output) / (key_output.shape[-1] ** (1/2))

        if mask : attention = np.tril(attention)

        weights = scipy.special.softmax(attention , axis = 1)

        output = weights * value_output

        return output , weights
    
class Encoder_Block:

    def __init__(self , model_dim):

        self.model_dim = model_dim
        
        self.mhsa = mhsa(self.model_dim)
        self.linear_ = linear(self.model_dim , self.model_dim)

        self.parameters = [self.mhsa.parameters]

    def forward(self , inps):

        attention , weights = self.mhsa.forward(inps , inps, inps)

        attention = layer_norm(attention)

        linear_attention = self.linear_.forward(attention)
        attention = layer_norm(linear_attention + attention)

        return attention , weights
    
class Encoder:

    def __init__(self , num_blocks , vocab_size , max_seq_len , model_dim):

        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        self.tok_embed = embedding(self.vocab_size , self.model_dim)
        self.pos_embed = embedding(self.vocab_size , self.model_dim)

        self.blokcs = []
        self.parameters = [self.tok_embed.parameters , 
                           self.pos_embed.parameters]
        
        for _ in range(self.num_blocks):

            obj = Encoder_Block(self.model_dim)

            self.blokcs.append(obj)
            self.parameters.append(obj.parameters)

    def forward(self , inps , mask = None):

        if len(inps.shape) == 1: 

            inputs = padding(inps , self.max_seq_len)
            pos_vals = np.array([val for val in range(self.max_seq_len)])

            tok_embeds = self.tok_embed.forward(inputs)
            pos_embeds = self.pos_embed.forward(pos_vals)

            embeds = tok_embeds + pos_embeds

            for block in self.blokcs :

                embeds , weights = block.forward(embeds)

                return embeds , weights
        
        elif len(inps.shape) == 2:

            inputs = np.empty(shape = (inps.shape[0] , self.max_seq_len))

            for index in range(inps.shape[0]):

                inputs[index] = padding(inps[index] , self.max_seq_len)

            index_vals = np.array([[val for val in range(self.max_seq_len)]
                                   for _ in range(inps.shape[0])])
            
            tok_embeds = self.tok_embed.forward(inputs)
            pos_embeds = self.pos_embed.forward(index_vals)

            embeds = tok_embeds + pos_embeds

            for block in self.blocks:

                embeds , weights = block.forward(embeds)

            return embeds , weights
        
        elif len(inps.shape) == 3 : 

            r_embeds = []
            r_weights = []

            for batch in inps:

                inputs = np.empty(shape = (batch.shape[0] , self.max_seq_len))

                for index in range(inps.shape[0]):

                    inputs[index] = padding(batch[index] , self.max_seq_len)

                index_vals = np.array([[val for val in range(self.max_seq_len)]
                                       for _ in range(batch.shape[0])])
                
                tok_embeds = self.tok_embed.forward(inputs)
                pos_embeds = self.pos_embed.forward(index_vals)

                embeds = tok_embeds + pos_embeds

                for block in self.blocks:

                    embeds , weights = block.forward(embeds)

                r_embeds.append(embeds)
                r_weights.append(weights)

            r_embeds = np.stack(r_embeds)
            r_weights = np.stack(r_weights)

        else : raise ValueError(f'Cannot Porcess inputs of shape {inps.shape}')


