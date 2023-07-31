"""
这是一个在3090上测试tgs的程序
"""
import os
import sys
import argparse
import torch

from collie.config import CollieConfig
from collie.data import CollieDatasetForTraining
from collie.models.llama.model import LlamaForCausalLM
from collie.models.moss_moon import Moss003MoonForCausalLM
from collie.utils.monitor import TGSMonitor, LossMonitor
from collie.controller.trainer import Trainer
from collie.module import GPTLMLoss
from collie.utils.dist_utils import setup_distribution

from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from datasets import load_dataset
from datasets import load_from_disk

try:
    # from rich.traceback import install
    # install()
    # decapoda-research/llama-7b-hf
    # openlm-research/open_llama_3b
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, default="openlm-research/open_llama_3b")
    arg_parser.add_argument('--tp_size', type=int, default=1)
    arg_parser.add_argument('--pp_size', type=int, default=1)
    arg_parser.add_argument('--batch_size', type=int, default=1)
    arg_parser.add_argument('--zero3', type=int, default=0)
    arg_parser.add_argument('--zeroplus', type=int, default=0)
    arg_parser.add_argument("--use_flash", type=int, default=1)
    arg_parser.add_argument("--grad", type=int, default=1)
    arg_parser.add_argument("--gpu", type=int, default=4)
    args = arg_parser.parse_args()

    map = {
        "openlm-research/open_llama_3b": 1,
        "decapoda-research/llama-13b-hf": 4,
        "decapoda-research/llama-7b-hf": 3,
        "decapoda-research/llama-30b-hf": 6,
        "decapoda-research/llama-65b-hf": 7,
        "./config/llama30b.json": 6,
        "./config/llama65b.json": 7,
        "./config/llama13b.json": 4,
        "./config/llama7b.json": 3,
        "./config/open3b.json": 1
    }

    data_path = './data'
        
    tag = f"p-hf-m{map[args.model_name]}bs{args.batch_size}tp{args.tp_size}pp{args.pp_size}grad{args.grad}gpu{args.gpu}_"
    if args.use_flash:
        tag += "flash"
    if args.zero3:
        tag += 'zero3'
    elif args.zeroplus:
        tag += 'zeroplus'

    config = CollieConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.tp_size = args.tp_size
    config.dp_size = 1
    config.pp_size = args.pp_size
    config.train_epochs = 1
    config.train_micro_batch_size = args.batch_size
    config.use_flash = args.use_flash
    config.gradient_accumulation_steps = int(args.grad)
    config.pp_partition_method = "uniform"
    config.ds_config = {
        "fp16": {
            "enabled": True
        },
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "monitor_config": {
            "enabled": True,
            "tag": tag,
            "csv_monitor": {
                "enabled": True,
                "output_path": "./results/"
            }
        }
    }

    if args.zero3:
        config.ds_config.update(
            {
                "zero_optimization": {
                    "stage": 3
                }
            }
        )
    elif args.zeroplus:
        config.ds_config.update({
            "zero_optimization": {
                "zero_quantized_weights": True,
                "zero_hpz_partition_size": 4,
                "zero_quantized_gradients": True
            }
        })

    config.ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    dschf = HfDeepSpeedConfig(config.ds_config)
    setup_distribution(config)
        
    tokenizer_path = "tokenizer.model"
    if "b" in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # "NeelNanda/pile-10k"
    dataset = [
        {
            "text": f"{sample['text']}"
        } for sample in load_from_disk("./data/")
    ][:10000]
    dataset = CollieDatasetForTraining(dataset=dataset, tokenizer=tokenizer, max_length=2048, seed=42)

    if "llama" in args.model_name:
        model = LlamaForCausalLM.from_config(config)
    elif "moss" in args.model_name:
        model = Moss003MoonForCausalLM.from_config(config)
    else:
        hfconfig = config.model_config
        hfconfig.gradient_checkpointing = True
        model = AutoModelForCausalLM.from_config(hfconfig)


    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    monitors = [TGSMonitor(config), LossMonitor(config)]

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        config = config,
        train_dataset = dataset,
        monitors = monitors,
        loss_fn = GPTLMLoss(-100)
    )
    trainer.train()

# torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=8 tgs_strategies.py --gpu 8 --batch_size 1 --grad 16 --zero3 1 --use_flash 0 --model_name decapoda-research/llama-7b-hf
# srun -p llm --quotatype=reserved --ntasks=16 --ntasks-per-node=8 python tgs_strategies.py --gpu 16 --batch_size 128 --tp_size 16

# PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29882 --nnodes=1 --nproc_per_node=8 main.py --gpu 8 --batch_size 1 --grad 16 --zero3 1 --use_flash 0 --model_name ./config/open3b.json

except Exception as e:
    print(e)
    with open('err.log', mode="a+") as file:
        file.write(f"{e}\n")

# PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29882 --nnodes=1 --nproc_per_node=8 test.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py --gpu 8 --batch_size 1 --grad 16 --zero3 1 --use_flash 0 --model_name ./config/open3b.json
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29882 --nnodes=1 --nproc_per_node=4 main.py --gpu 4 --batch_size 1 --grad 32 --zero3 1 --use_flash 0 --model_name ./config/open3b.json