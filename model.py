import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, vocab_size : int , d_model : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model) #multiplying with embedding vector dim as mentioned in the paper
    
    
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model : int, seq_length : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        #Creating a matrix of shape (seq_length , d_model) (row,column)
        pe = torch.zeros(seq_length, d_model)
    
        # Create a vector of shape (seq_length,1)
        position = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
        division_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model)) #arange generates even indices(2i)
        
        #Applying sin to even positions and cos to odd positions
        pe[:,0::2] = torch.sin(position * division_term)
        pe[:,1::2] = torch.cos(position * division_term)
    
        #Expanding the dimension of pe at 0th index to make dim (1,seq_length,d_model)
        pe = pe.unsqueeze(0)
        
        #Saving pe as buffer, as buffers are not model parameters(they cannot be optimised during training) and it is saved with the model
        self.register_buffer('pe', pe)
        
    def forward(self,x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x) #Requires_grad(False) as positional encodings are not model parameters
    
    
class LayerNormalisation(nn.Module):
    def __init__(self, eps : float = 10**-6): #eps is set so small, so division term for normalisation doesn't become zero
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplication term which are learnable parameters
        self.bias = nn.Parameter(torch.zeros(1)) #Addition term which are also learnable parameters
        
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * ((x - mean)/(std + self.eps)) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int , d_ff : int, dropout : float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias = True) #For W1 and B1 where d_model is input feature and d_ff is output feature
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias = True) # For W2 and B2 where d_ff is input feature and d_model is output feature
        
    def forward(self, x):
        #(Batch_size, seq_length, d_model) --> Apply linear to get --> (Batch_size, seq_length,d_ff) --> Apply linear to get back --> (Btahc_size,seq_length, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model : int, h : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h ==0, "d_model is not divisible by h " #assertion for checking dk = (d_model/h) so that there is no mismatch in shape
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #W_q matrix
        self.w_k = nn.Linear(d_model, d_model) #W_k matrix
        self.w_v = nn.Linear(d_model, d_model) #W_v matrix
        
        self.w_o = nn.Linear(self.d_k*h, d_model) #W_o matrix
        self.dropout = nn.Dropout(dropout)
        
        
    @staticmethod # This makes the module inside a class behave like a regular function. IT BELONGS TO THE CLASS, BUT NOT AN INSTANCE
    def attention(query, key, value, mask,dropout = nn.Dropout):
        d_k = query.shape[-1]
        
        # (batch, h, seq_length, d_k) --> (batch, h ,seq_length, seq_length)
        attention_scores = query @ key.transpose(-1,-2) / math.sqrt(d_k) # @ is matrix multiplication in pytorch
        if mask is not None : #Checks whether mask is not None that is whether a mask tensor is provided. AND NOT THE VALUES IN MASK
            
            # mask => if mask =1(encoder) , then it allows attention if it is 0(first multi head attention), it blocks attention so that the FUTURE TOKENS ARE NOT SEEN
            attention_scores.masked_fill_(mask == 0, -1e9) #mask_fill_ checks for all mask ==0 , it is assigned with a very large negative no. -1e9

        attention_scores = attention_scores.softmax(dim = -1) #Applies softmax function only to the last dimension , (batch,h,seq_length,d_k)
        
        if dropout is not None :
            attention_scores = dropout(attention_scores)
            
        return (attention_scores @ value), attention_scores # attention_scores @ value for correct matrix multiplication order
    
    def forward(self,q,k,v,mask): # * means element wise multiplication of tensor and self.w_q(q) means matrxi multiplication
        query = self.w_q(q)  #Matrix multiplication of q with w_q
        key = self.w_k(k)    #Matrix multiplication of k with w_k
        value = self.w_v(v)  #Matrix multiplication of v with w_v
        
        # In the above we are going from (batch_size, seq_length, d_model) --> (batch_size,seq_length,d_model) after matrix multiplication
        
        # Since we need attention of each h heads so reshaping the tensor and swaping 1st index with 2nd index and vice-versa
        #(batch, seq_length, d_model) --> (batch, seq_length, h ,d_k ) --> (batch,h, seq_length, d_k) => done for all q,k,v
        #Taking transpose so attention is applied to all h heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch,h, seq_length, d_k) --> (batch, seq_length , h, d_k) --> (batch, seq_length, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  #Concatenating all heads
        
        # (batch, seq_length, d_model) --> (batch, seq_length ,d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation() #Since the residual connections are given to layernormalosation
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # residual connection. sublayer can be multi-head attention or feed forawrd network
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
       super().__init__()
       self.self_attention_block = self_attention_block
       self.feed_forward_block = feed_forward_block
       self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers # These are the multiple layers executed by the Encoder Block class and is stored as layers
        self.norm = LayerNormalisation() # Last layer's normalisation
        
    def forward(self,x,mask):
        for layer in self.layers:#Each and every layer(The encoder block is executed N times outputing N layers) is masked
            x = layer(x,mask) # This basically calls the forward method of EncoderBlock , so each layers till N layers, the EncoderBlock is applied
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock,cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) # range is 3 because there are 3 residual connections in decoder block
        
    def forward(self, x, encoder_output, src_mask, target_mask): # src_mask is the source mask used for ignoring the padded tokens from the encoder output and target_mask is the mask used for preventing to see the future training examples
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()
        
    def forward(self, x ,encoder_output, src_mask, target_mask): 
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask) # This calls the forward method of DecoderBlock class , so that the DecoderBLock is applied to each of the N layers
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (Batch, seq_length, d_model) --> (Batch, seq_length, vocab_size) ==> Basically converting the output embeddings(of size d_model = 512) back to vocabulary size
        return torch.log_softmax(self.proj(x), dim = -1) # logsoftmax is used to give better results and logsoftmax is applied to the last dimension
    
class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : InputEmbeddings, target_embed : InputEmbeddings, src_pos : PositionalEncoding, target_pos : PositionalEncoding, projection_layer = ProjectionLayer):
        super().__init__() # Source embeddings and target embeddings are defined seperately as we have to get embeddings for two different languages, same goes with positional encoding
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask): #here src is the input language and src_mask is the mask of the encoder
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output,src_mask, target,  target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size : int, target_vocab_size : int, src_seq_len : int, target_seq_len : int, d_model : int = 512, N : int = 6, h : int = 8, dropout : float = 0.1, d_ff = 2048):
    # Since there are a lot of hyperparameters, we are writing this build_transformer function, also vocab size of src and target can be different as we use two different languages, so sentence length may change
    
    #Create embedding layers
    src_embed = InputEmbeddings(src_vocab_size, d_model)
    target_embed = InputEmbeddings(target_vocab_size, d_model)
    
    #Create positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)
    
    # Create encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block =  EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model ,h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #Create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)
    
    #Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, target_embed ,src_pos, target_pos, projection_layer)
    
    #Intialise parameters(not random but good)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer