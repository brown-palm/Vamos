import torch
from .base_dataset import BaseDataset
import pandas as pd
import json

class PTest(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/ptest/{split}.csv')
        if args.use_cap:
            if args.cap_model == '13b':
                self.caption = json.load(open(f'./data/ptest/llava_13b_n6.json'))
            elif args.cap_model == '7b':
                self.caption = json.load(open(f'./data/ptest/llava_7b_n6.json'))
        self.features = torch.load(f'./data/ptest/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)'}
        self.num_options = 3
        self.use_cap = args.use_cap
        print("use cap", args.use_cap)
        print("use vis", args.use_vis)
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]
        vid = self.data['video'].values[idx]
        qid = self.data['qid'].values[idx]
        # find caption where video id is vid
        caption = ""
        if self.use_cap:
            caption = self.caption[vid].split("The video shows: ")[-1].strip()
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'c_text': caption, 'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        video, video_len = self._get_video(f'{vid}')
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": -1}

    def __len__(self):
        return len(self.data)