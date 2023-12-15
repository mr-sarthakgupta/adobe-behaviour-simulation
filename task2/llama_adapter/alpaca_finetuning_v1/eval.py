import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path

import models_llama_adapter
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from engine_finetuning import train_one_epoch, val_one_epoch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from transformers import (
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
)


from llama import Tokenizer

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, data_path, model_path, max_words=30, partition="train"):
        self.ann = json.load(open(data_path))
        data_length = len(self.ann)
        if partition == "val":
            self.ann = self.ann[:1000]
        if partition == "train1":
            self.ann = self.ann[400: (data_length - 400) // 4 + 400]
        if partition == "train2":
            self.ann = self.ann[(data_length - 400) // 4 + 400: (data_length - 400) // 2 + 400]
        if partition == "train3":
            self.ann = self.ann[(data_length - 400) // 2 + 400: (data_length - 400) * 3 // 4 + 400]
        if partition == "train4":
            self.ann = self.ann[(data_length - 400) * 3 // 4 + 400:(data_length - 400) * 3 // 4 + 1400]

        self.max_words = max_words
        tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        # example = prompt + str(ann["output"])
        # if len(self.tokenizer1.encode(example, bos=True, eos=True)) > self.max_words:
        
        prompt = torch.tensor(self.tokenizer1.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        prompt_padding = 300 - prompt.shape[0]
        if prompt_padding > 0:
            prompt = torch.cat((prompt, torch.zeros(prompt_padding, dtype=torch.int64) - 1))
        if prompt_padding < 0:
            prompt = prompt[:300]
        example = torch.cat((prompt, torch.tensor(self.tokenizer1.encode(str(ann["output"]), bos=True, eos=True), dtype=torch.int64)), dim = 0)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="./llama", type=str, help="path of llama model")
    parser.add_argument("--model", default="llama7B_adapter", type=str, metavar="MODEL", help="Name of model to train")

    parser.add_argument("--adapter_layer", type=int, default=30, metavar="LENGTH", help="the number of adapter layer")

    parser.add_argument("--adapter_len", type=int, default=10, metavar="LENGTH", help="the adapter length")

    parser.add_argument("--max_seq_len", type=int, default=512, metavar="LENGTH", help="the maximum sequence length")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR")

    # Dataset parameters
    parser.add_argument("--data_path", default="/instruction_dataset/", type=str, help="dataset path")

    parser.add_argument("--output_dir", default="./output_dir", help="path where to save, empty for no saving")
    parser.add_argument("--log_dir", default="./output_dir", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser


def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train1 = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="train1"
    )
    dataset_train2 = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="train2"
    )
    dataset_train3 = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="train3"
    )
    dataset_train4 = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="train4"
    )
    dataset_val = InstructionDataset(
        data_path=args.data_path, model_path=args.llama_model_path, max_words=args.max_seq_len, partition="val"
    )

    # print(dataset_train)
    # print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train1 = torch.utils.data.DistributedSampler(
            dataset_train1, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train2 = torch.utils.data.DistributedSampler(
            dataset_train2, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train3 = torch.utils.data.DistributedSampler(
            dataset_train3, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train4 = torch.utils.data.DistributedSampler(
            dataset_train4, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        # print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train1 = torch.utils.data.RandomSampler(dataset_train1)
        sampler_train2 = torch.utils.data.RandomSampler(dataset_train2)
        sampler_train3 = torch.utils.data.RandomSampler(dataset_train3)
        sampler_train4 = torch.utils.data.RandomSampler(dataset_train4)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train1 = torch.utils.data.DataLoader(
        dataset_train1,
        sampler=sampler_train1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2,
        sampler=sampler_train2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_train3 = torch.utils.data.DataLoader(
        dataset_train3,
        sampler=sampler_train3,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_train4 = torch.utils.data.DataLoader(
        dataset_train4,
        sampler=sampler_train4,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_llama_adapter.__dict__[args.model](args)
    model.to(device)
    tokenizer = Tokenizer(model_path=args.llama_model_path + "./tokenizer.model")

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train1.sampler.set_epoch(epoch)
            data_loader_train2.sampler.set_epoch(epoch)
            data_loader_train3.sampler.set_epoch(epoch)
            data_loader_train4.sampler.set_epoch(epoch)
            data_loader_val.sampler.set_epoch(epoch)
        # if epoch == 0 or epoch == 4:
        #     dataloader = data_loader_train1
        # elif epoch == 1:
        #     dataloader = data_loader_train2
        # elif epoch == 2:
        #     dataloader = data_loader_train3
        # elif epoch == 3:
        #     dataloader = data_loader_train4
        
        # train_stats = train_one_epoch(
        #     model, tokenizer, dataloader, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        # )

        val_stats = val_one_epoch(
            model, tokenizer, data_loader_train4, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            print('meow meow')

        # log_stats = {
        #     **{f"train_{k}": v for k, v in train_stats.items()},
        #     "epoch": epoch,
        #     **{f"val_{k}": v for k, v in val_stats.items()},
        # }

        # if args.output_dir and misc.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        # break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    instance_ids = ['your_instance_id_here']  # Replace with your actual instance ID(s)
    ec2.stop_instances(InstanceIds=instance_ids)

if __name__ == "__main__":

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    print('everything went well')
