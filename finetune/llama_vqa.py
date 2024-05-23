import torch
import json
from llama import ModelArgs, Tokenizer_llama, Tokenizer_llama3, Transformer
from pathlib import Path

def LLaMA_VQA(args, **kwargs):
    with open(f'{args.llama_model_path}{args.model}/params.json', "r") as f:
        params = json.loads(f.read())
    if 'llama3' in args.llama_model_path:
        tokenizer = Tokenizer_llama3(model_path=f'{args.llama_model_path}/tokenizer.model')
    else:
        tokenizer = Tokenizer_llama(model_path=f'{args.llama_model_path}/tokenizer.model')
    print(f"Using model: {args.model}")
    
    
    checkpoints = (Path(args.llama_model_path) / args.model).glob("*.pth")
    checkpoints = sorted(checkpoints)
    
    loaded = []
    for x in checkpoints:
        print("loading from", x)
        loaded.append(torch.load(x, map_location="cpu"))
    
    if len(loaded) == 1:
        full_state_dict = loaded[0]
    else:
        full_state_dict = {}
        split_dims = {}
        
        def add_weight_with_split_dim(name, dim):
            if dim < 0:  # bcast without split
                full_state_dict[name] = loaded[0][name].clone()
            else:
                full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
            for x in loaded:
                del x[name]
            split_dims[name] = dim
        
        add_weight_with_split_dim("tok_embeddings.weight", 1)
        add_weight_with_split_dim("norm.weight", -1)
        add_weight_with_split_dim("output.weight", 0)
        for i in range(params["n_layers"]):
            print("gathering layer %d of %d" % (i, params["n_layers"]))
            layer_prefix = f"layers.{i}."
            bcast_names = ["attention_norm.weight", "ffn_norm.weight"]
            column_parallel_names = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "feed_forward.w1.weight", "feed_forward.w3.weight"]
            row_parallel_names = ["attention.wo.weight", "feed_forward.w2.weight"]
            for key in bcast_names:
                add_weight_with_split_dim(layer_prefix + key, -1)
            for key in column_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 0)
            for key in row_parallel_names:
                add_weight_with_split_dim(layer_prefix + key, 1)
    

    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, max_batch_size=32, adapter_len=args.adapter_len, adapter_layer=args.adapter_layer, **params)
    
    if "llava" in args.llama_model_path:
        model_args.vocab_size = 32064
    else:
        model_args.vocab_size = tokenizer.n_words
    print("vocab size", model_args.vocab_size)
    print("precision", args.precision)
    if args.precision == "fp16":
        torch.set_default_dtype(torch.half)
    elif args.precision == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    elif args.precision == "fp32":
        torch.set_default_dtype(torch.float)

    model_llama_vqa = Transformer(model_args, args)
    torch.set_default_dtype(torch.float)
    missing_keys, unexpected_keys = model_llama_vqa.load_state_dict(full_state_dict, strict=False)

    for name, param in model_llama_vqa.named_parameters():
        if ('gate' in name) or ('adapter' in name) or ('temporal_emb' in name) or ('visual_proj' in name) or ('selector' in name):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False
        print(name, param.requires_grad)

    return model_llama_vqa