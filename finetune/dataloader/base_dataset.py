import torch
from torch.utils.data import Dataset
import copy

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split):
        self.args = args
        self.max_feats = args.max_feats
        self.features_dim = 768
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.split = split
    
    def _get_padding_id(self, text_id):
        padding_text_id = torch.zeros((len(text_id), self.max_seq_len), dtype=torch.int64) - 1
        for i, tid in enumerate(text_id):
            padding = self.max_seq_len - len(tid)
            if padding >= 0:
                padding_text_id[i, :len(tid)] = tid
            else:
                padding_text_id[i] = tid[:self.max_seq_len]
                print('max sequence length overflow')
        return padding_text_id
    
    def _get_text_token(self, text, answer):
        vqa_id, vqa_prefix_index, vqa_video_start = self.tokenizer.encode_vqa(text=text, max_feats=self.max_feats, split=self.split, answer_mapping=self.answer_mapping, answer=answer)
        vqa_id = [torch.tensor(v_id, dtype=torch.int64) for v_id in vqa_id]
        vqa_padding_text_id = self._get_padding_id(vqa_id)

        # label
        vqa_label = copy.deepcopy(vqa_padding_text_id)
        vqa_label[:, :vqa_prefix_index] = -1
        vqa_label_mask = vqa_label.ge(0)
        vqa_label[~vqa_label_mask] = 0
        vqa_label_mask = vqa_label_mask.float()  
                
        # text mask
        vqa_text_mask = vqa_padding_text_id.ge(0)
        vqa_padding_text_id[~vqa_text_mask] = 0
  
        # video index
        vqa_video_index = torch.arange(vqa_prefix_index, vqa_prefix_index + self.max_feats)
        
        text_id = {'vqa': vqa_padding_text_id}
        label = {'vqa': vqa_label}
        video_start = {'vqa': vqa_video_start}
        video_index = {'vqa': vqa_video_index}
        label_mask = {'vqa': vqa_label_mask}
        return text_id, label, video_start, video_index, label_mask