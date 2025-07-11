
##Training

For reasons I cannot discern you will need to run all of the below to get it to work

```
pip install -r requirements
pip install transformers, trl, dotenv
rm -rf /workspace/gemma/cyberkind/wandb
pip install wandb
```

You will also need to setup accelerate config. (Passing the file doesn't seem to work)

Run training with the following if your GPU is large enough (~48GB VRAM).

```
accelerate launch --config_file accelerate_config.yaml training/behavior_llm_sft.py
```

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