import torch
from util import misc
from .nextqa import NextQA
from .ptest import PTest
from .intentqa import IntentQA


dataset_mapping = {'nextqa': NextQA, 'intentqa': IntentQA, 'ptest': PTest}
num_options_mapping = {'nextqa': 5, 'intentqa': 5, 'ptest': 3}

def load_data(args, tokenizer, split='train'):
    args.num_options = num_options_mapping[args.dataset]
    dataset = dataset_mapping[args.dataset](args=args, tokenizer=tokenizer, split=split)
    
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_collate,
                                              pin_memory=args.pin_mem, drop_last=False)

    return data_loader

def batch_collate(batch):
    bs = len(batch)
    vid = [batch[i]["vid"] for i in range(bs)]
    video = torch.stack([batch[i]["video"] for i in range(bs)])
    video_len = torch.tensor([batch[i]["video_len"] for i in range(bs)], dtype=torch.long)
    text = [batch[i]["text"] for i in range(bs)]
    qid = [batch[i]["qid"] for i in range(bs)]
    qtype = torch.tensor([batch[i]['qtype'] for i in range(bs)])
    
    vqa_id = torch.stack([batch[i]['text_id']['vqa'] for i in range(bs)])
    text_id = {'vqa': vqa_id}
    
    vqa_label = torch.stack([batch[i]['label']['vqa'] for i in range(bs)])
    label = {'vqa': vqa_label}
    
    vqa_video_start = [batch[i]["video_start"]['vqa'] for i in range(bs)]
    video_start = {'vqa': vqa_video_start}

    vqa_video_index = torch.stack([batch[i]["video_index"]['vqa'] for i in range(bs)])
    video_index = {'vqa': vqa_video_index}
    
    vqa_label_mask = torch.stack([batch[i]["label_mask"]['vqa'] for i in range(bs)])
    label_mask = {'vqa': vqa_label_mask}

    answer = torch.tensor([batch[i]["answer"] for i in range(bs)])
    return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
            "video_index": video_index, "label_mask": label_mask, "qid": qid, "answer": answer, "qtype": qtype}