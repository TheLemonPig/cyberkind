
##Training

If you are in runpod, I would recommend torch==2.4.0 template over the 2.8.0 one
(torch==2.8.0 is not yet stable)

For reasons I cannot discern you will need to run all of the below to get it to work

```
pip install -r requirements.txt
pip install transformers
pip install trl
pip install dotenv
pip install -U bitsandbytes
rm -rf /workspace/gemma/cyberkind/wandb
pip install wandb
```

You will also need to setup accelerate config. (Passing the file doesn't seem to work)

Answer the questions as visible here:

```
---------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                                       
---------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                               
multi-GPU                                                                                                                                          
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                         
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes                 
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                  
Do you want to use DeepSpeed? [yes/NO]: NO                                                                                                         
Do you want to use FullyShardedDataParallel? [yes/NO]: NO                                                                                          
Do you want to use Megatron-LM ? [yes/NO]: NO                                                                                                      
How many GPU(s) should be used for distributed training? [1]:4                                                                                     
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:                                                  
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: yes                                             
---------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
no  
```

Run training with the following if your GPU is large enough (~48GB VRAM).

```
accelerate launch --config_file accelerate_config.yaml training/behavior_llm_sft.py
```

or

```
accelerate launch training/behavior_llm_sft.py
```

considering the yaml isn't working.

If your GPU is not big enough but you have multiple GPUs, then uncomment line 69 in `behavior_llm_sft.py` and run the file as you normally would. This will shard the model.

##Additional troubleshooting

Make sure to run the following to fix a bug I have not yet caught at the source:



Please also note that the `requirements.txt` has a few runpod-specific packages. To configure for other environments, focus on replacing all GPU-related packages and reinstalling locally.

For example replace the following with publically available versions of these packages:
```
torch==2.8.0.dev20250319+cu128
torchaudio==2.6.0.dev20250319+cu128
torchvision==0.22.0.dev20250319+cu128
```

If in doubt, I would recommend building your own requirements.txt 

If you need to push to git you will need to input these lines first:
```
git config --global user.email "lemontartpig@gmail.com"
git config --global user.name "TheLemonPig"
```