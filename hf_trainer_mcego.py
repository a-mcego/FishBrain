#%%
import time
import multiprocessing
import sys

if len(sys.argv) < 2:
    print(f"{sys.argv[0]} scratch/load")
    exit(0)

import numpy as np
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0

from metrics.accuracy import Tester
from data.processors import process_sample, scorer
from data.tokenizer import Tokenizer
from model.conv_bigmega_v0.model import ConvTransformer


BATCHSIZE = 1024
N_CHECKPOINT = 2500
D_EMB = 256
N_LAYERS = 8
N_HEADS = 8
CALCULATE_ACCURACY = True
LOADSTATE = (sys.argv[1] == "load")
device = 'cuda'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    """ 
    ---------------------------------------------
    load dataset, initialize tokenizer and tester
    ---------------------------------------------
    """
    
    tokenizer = Tokenizer()
    tester = Tester(batchsize=BATCHSIZE, tokenizer=tokenizer)

    dataset = load_dataset(path="mauricett/lichess_sf",
                           split="train",
                           streaming=True, trust_remote_code=True)

    dataset = dataset.map(function=process_sample, 
                          fn_kwargs={"tokenizer": tokenizer, "scorer": scorer})

    dataloader = DataLoader(dataset, 
                            batch_size=BATCHSIZE,
                            num_workers=4,
                            prefetch_factor=8)


    """
    ---------------------------------------------
    init model
    ---------------------------------------------
    """
    model = ConvTransformer(D_EMB, N_LAYERS, N_HEADS)
    model = model.to(device)
    model_dict = {'acc': np.zeros((1, 62, 100)),
                  'steps': [0],
                  'loss': [],
                  'grad_norm': [],
                  'batchsize': []}

    scaler = torch.cuda.amp.GradScaler(init_scale=2**16, growth_factor=1.5, backoff_factor=0.66)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    bce_loss = nn.BCEWithLogitsLoss()
    max_norm = 5
    
    #%%
    n_steps = 0
    n_epochs = 100
    timer = time.perf_counter()

    if LOADSTATE:
        model.load_state_dict(torch.load("model/fishweights_sqrelu.pt"))
        model_dict = torch.load("model/model_dict_sqrelu.pt")
        scaler.load_state_dict(torch.load("model/scaler_dict_sqrelu.pt"))
        optimizer.load_state_dict(torch.load("model/optimizer_sqrelu.pt"))
        n_steps = model_dict['steps'][-1]
        
    

    for epoch in range(n_epochs):
        print("Epoch %i" % epoch)

        dataset = dataset.shuffle()

        for batch in dataloader:
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                x = model(batch['fens'], batch['moves'])
                scores = batch['scores'].to(device)
                loss = bce_loss(x, scores)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            n_steps += 1

            model_dict['grad_norm'].append(float(grad_norm.cpu()))
            model_dict['loss'].append(loss.item())

            print(f"{n_steps}       ", end="\r")

            if not (n_steps % N_CHECKPOINT):
                speed = (N_CHECKPOINT * BATCHSIZE) / (time.perf_counter() - timer)

                if CALCULATE_ACCURACY:
                    accuracy = tester(model)
                    model_dict['acc'] = np.concatenate([model_dict['acc'], accuracy])
                    model_dict['steps'].append(n_steps)
                    model_dict['batchsize'].append(BATCHSIZE)
                    print("%i: %.1f accuracy, %i positions / s" % (n_steps, model_dict['acc'][-1].mean() * 100, speed))
                else:
                    print("%i: %i positions / s" % (n_steps, speed))

                torch.save(model.state_dict(), 'model/fishweights_sqrelu.pt')
                torch.save(optimizer.state_dict(), "model/optimizer_sqrelu.pt")
                torch.save(model_dict, "model/model_dict_sqrelu.pt")
                torch.save(scaler.state_dict(), "model/scaler_dict_sqrelu.pt")
                
                timer = time.perf_counter()