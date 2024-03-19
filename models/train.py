#! /usr/bin/env python3
from emoe_v1 import ElasticMoELlamaForCausalLM
from emoe_config import ElasticMoELlamaConfig
from emoe_utils import profile, writing
from packed_dataset import PackedDataset

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

# from datasets import load_dataset

import os
import glob
from pickle import dump
import cProfile
import multiprocessing
from jsonargparse import CLI


PROC_COUNT = multiprocessing.cpu_count()

def train(
        # Trace
        profiling_path:str="profile", 
        tensorboard_path:str="tensorboard",
        logging_step_interval:int=1,

        # Schedule
        epochs:int=1,
        warmup_steps:int=0,
        max_steps:int=100,
        eval_step_interval:int=1000,
        checkpoint_path:str="checkpoint",
        checkpoint_step_interval:int=3000,

        # Data
        data_cache_dir:str="data",
        block_size:int=10,
        micro_batch_size:int=1,
        batch_size:int=1,
        chunks=1,
        num_workers:int=4,

        # Optimizer
        learning_rate:float=1e-4,
        beta1:float=0.9,
        beta2:float=0.95,
        weight_decay:float=0.01,
        decay_learning_rate:bool=True,
        gradient_clip:float=1.0,
):
    """
    Train a model using the Elastic MoE Llama model.
    """

    train_path_str = os.path.join(data_cache_dir, "red_pajama_sample",  "train/*")
    eval_path_str = os.path.join(data_cache_dir, "red_pajama_sample",  "validation/*")
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)
    with writing(tensorboard_path) as writer:

        # Begin Model
        torch.cuda.memory._record_memory_history(
            True,
            # pre-snapshot max alloc/free events = 100,000
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )
        writer.add_text('Model Event', "Model initializing.")
        config = ElasticMoELlamaConfig()
        model = ElasticMoELlamaForCausalLM(config).to("cuda")
        criterion = torch.nn.CrossEntropyLoss() # unused b/c HF provides it
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2), 
            weight_decay=weight_decay, 
            foreach=False
        )
        def lr_scheduler(step):
            lr_scale = 1.0
            writer.add_scalar("Scheduler LR Update", lr_scale, step)
            return lr_scale
        scheduler = LambdaLR(optimizer, lr_scheduler)
        writer.add_text('Model Event', "Model initialized.")
        # End Model
        
        # Begin Checkpoint Loading
        global_step = 0
        epoch = 0
        train_begin_slice = 0
        train_end_slice = max_steps
        eval_begin_slice = 0
        eval_end_slice = max_steps # eval should be static but we'll just do the same as train for now
        writer.add_text('Checkpoint Event', f"Loading checkpoint from '{checkpoint_path}/' if it exists.")
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoints = os.listdir(checkpoint_path)
            latest = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])
            checkpoint_filename = os.path.join(checkpoint_path, f'checkpoint_{latest}.pt')
            if os.path.exists(checkpoint_filename):
                checkpoint = torch.load(checkpoint_filename)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                global_step = checkpoint['global_step']
                train_begin_slice = checkpoint['train_begin_slice']
                train_end_slice = train_begin_slice + max_steps
                eval_begin_slice = checkpoint['eval_begin_slice']
                eval_end_slice = eval_begin_slice + max_steps
                epoch = checkpoint['epoch']
                writer.add_text('Checkpoint Event', f"Loaded checkpoint at {global_step} from '{checkpoint_filename}'. Training to {max_steps} steps.")
        # End Checkpoint Loading

        # Begin Data
        writer.add_text('Data Event', "Initializing data loaders.")
        train_filenames = sorted(glob.glob(train_path_str))[train_begin_slice:train_end_slice]
        training_dataset = PackedDataset(train_filenames, n_chunks=chunks, block_size=block_size, shuffle=False)
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

        eval_filenames = sorted(glob.glob(eval_path_str))[eval_begin_slice:eval_end_slice]
        eval_dataset = PackedDataset(eval_filenames, n_chunks=chunks, block_size=block_size, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        writer.add_text('Data Event', "Initialized data loaders.")
        # End Data

        # Begin Train
        writer.add_hparams(hparam_dict={
                # Optimization
                "lr":learning_rate,
                "beta1":beta1,
                "beta2":beta2,
                "weight_decay":weight_decay,
                "decay_learning_rate":decay_learning_rate,
                "gradient_clip":gradient_clip,
                
                # Data loading
                "training_dataset_path":train_path_str,
                "eval_dataset_path":eval_path_str,
                "num_workers":num_workers,
                "batch_size":batch_size,
                "micro_batch_size":micro_batch_size,
                "block_size":block_size,
                "chunk_size":chunks,

                # Training
                "epochs":epochs,
                "warmup_steps":warmup_steps,
                "max_steps":max_steps,
                "train_begin_slice":train_begin_slice,
                "train_end_slice":train_end_slice,
                "eval_begin_slice":eval_begin_slice,
                "eval_end_slice":eval_end_slice,
            },metric_dict={},global_step=global_step)

        model.train()
        for epoch in range(epoch, epochs):
            for step, td in enumerate(training_dataloader):
                input_ids = td[:, :].contiguous().to(model.device)
                labels = td[:, :].contiguous().to(model.device)

                with profile(f"forward"):
                    # HF implements target text shifting, or we can do it manually.
                    m_output = model(input_ids, labels=labels) 
                    loss = m_output.loss
                with profile(f"backward"):
                    loss.backward()
                with profile(f"optimizer {type(optimizer).__name__}"):
                    if gradient_clip is not None:
                        clip = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        clip_magnitude = torch.sum(clip)
                        writer.add_scalar('Gradient Clip Magnitude', clip_magnitude, step)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Step Complete
                global_step += 1
                if global_step >= max_steps:
                    break
                
                if step % logging_step_interval == 0:
                    writer.add_scalar('Training Loss', loss.item(), step)
                if step + 1 % eval_step_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        with profile(f'evaluation'):
                            for eval_step, eval_td in enumerate(eval_dataloader):
                                eval_inputs = eval_td[:, :].contiguous().to(model.device)
                                eval_targets = eval_td[:, :].contiguous().to(model.device)
                                eval_m_output = model(eval_inputs, labels=eval_targets)
                                eval_loss = eval_m_output.loss
                                writer.add_scalar('Evaluation Loss', eval_loss.item(), eval_step)
                    model.train()

                if step % checkpoint_step_interval == 0:
                    print("Checkpointing")
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'global_step': global_step,
                        'train_begin_slice': train_end_slice,
                        'epoch': epoch,
                    }
                    checkpoint_filename = os.path.join(checkpoint_path, f'checkpoint_{global_step}.pt')
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save(checkpoint, checkpoint_filename)
                    writer.add_text('Checkpoint Event', f"Checkpointed at {global_step} / {max_steps} with file at {checkpoint_filename}")
                if global_step >= max_steps:
                    writer.add_text('Reached Max Steps', f"Reached {global_step} / {max_steps}")
                    break
            scheduler.step()
        # End Train


        # Begin Cleanup
        def snap(profiling_path=None):
            if profiling_path is None:
                writer.add_text("Memory Snapshot", "No snapshot path provided, so we didn't write the profiling snapshot")
            if not os.path.exists(profiling_path):
                os.makedirs(profiling_path, exist_ok=True)
            snapshot_path = os.path.join(profiling_path, "snapshot.pkl")
            with open(snapshot_path, "wb") as f:
                snapshot = torch.cuda.memory._snapshot()
                dump(snapshot, f)
                writer.add_text("Memory Profiling Snapshot", f"Snapshot written to {snapshot_path}")
        if profiling_path is not None:
            snap(profiling_path)
        # End Cleanup



if __name__ == '__main__':
    CLI(train)