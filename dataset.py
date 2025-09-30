import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        
        self.ds = ds 
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, index):
            src_target_pair = self.ds[index]
            src_text = src_target_pair[self.src_lang]
            tgt_text = src_target_pair[self.tgt_lang]
            
            #Splitting sentences into tokens(words)
            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
            
            # Truncate tokens if too long, so we never raise an error
            if len(enc_input_tokens) > self.seq_len - 2:
                enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            if len(dec_input_tokens) > self.seq_len - 1:
                dec_input_tokens = dec_input_tokens[:self.seq_len - 1]

            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 for SOS and EOS
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 for SOS

            # Now we pass one sentence (eng) as input to the encoder,  one sentence (tamil) as input to the decoder and one sentence as label(expected output) to the decoder output
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
                ],
                dim=0,
            )
            #Add SOS to the decoder input 
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype = torch.int64),
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
                ],
                dim=0
            )
            #
            label = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype = torch.int64),
                    self.eos_token, 
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
                ],
            dim=0,
            )
            
            assert encoder_input.size(0) == self.seq_len
            assert decoder_input.size(0) == self.seq_len
            assert label.size(0) == self.seq_len
            
            return{
                "encoder_input" : encoder_input, #seq_len
                "decoder_input" : decoder_input, #seq_len
                "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len) => This encoder mask ignores padded tokens during attention mechnaism
                "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1,seq_len) & (1,1,seq_len) ==> This decoder mask ignores padded tokens during attention mechanism also it should not be able to see the future values, so they are replaced with a large negative value
                "label" : label,
                "src_text" : src_text,
                "tgt_text" :tgt_text
            }
            
def casual_mask(size):
    mask = torch.triu(torch.ones((1,size,size)), diagonal = 1).type(torch.int) #torch.triu returns 0 to all the words below the diagnol before the word , but we want the opposite of it, so return mask ==0
    return mask == 0
