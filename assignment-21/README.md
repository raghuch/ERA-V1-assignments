---
title: raghunc0nano-gpt-shakespeare-demo
app_file: app.py
sdk: gradio
sdk_version: 3.39.0
---
This is an example of a nano-gpt trained on mini-shakespeare text of size 1MB. The model follows the video exactly and was trained for 5000 iters. The training and the text generation code are in the [train_shakespeare.ipynb](./train_shakespeare.ipynb) file. To generate the text from model during inference time, run the following lines:

```
context = torch.zeros((1, 1), dtype=torch.long)
print(char_tokenizer.decode(m3.generate(context, max_new_tokens=500)[0].tolist()))
```

Here we start with a new "context" vector of zero tensor (standing in for "START" token) and  "max_new_tokens" is the max number of tokens (or letters here, in this demo) that will be generated. I have limited it to 500 to be able to inference on CPU in a reasonable time (around 10s) -- which is suitable for Huggingface gradio demo without payment. Inference on GPU can support max_new_tokens to any value; tested upto a few thousand. 

The model checkpoint is the 'nano_gpt_ckpts' dir. The hyper params used are the exact same shown in the video:
```
vocab_size=65, n_layer=6, n_head=6, n_embed=384, block_size=256,
                  bias=False, dropout=0.2
```

