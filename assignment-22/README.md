This repo is to show the results of training a Pythia 160M model with GPTNeoxMLP and Gelu. The target is to reduce the training loss below 3.5 which is observed in the [main.ipynb](./main.ipynb) file logs. The generate function is taken from lit-gpt [base generate](https://github.com/Lightning-AI/lit-gpt/blob/main/generate/base.py) function, where the temperature and max tokens can be set.

Samples of generated text are shows in the [tsai_gpt/generate.ipynb](./tsai_gpt/generate.ipynb) notebook. To generate text, run the command:

```
!python3 generate.py --prompt="there are torsion-free hyperbolic groups that uniformly " --checkpoint_dir=/home/raghu/work/ERA-V1-assignments/assignment-22/checkpoints/meta-llama/Llama-2-7b-chat-hf/ --max_new_tokens=250 --temperature=0.8 --num_samples=1
```
which gives the sample text:

```
there are torsion-free hyperbolic groups that uniformly 100,000 times a day. The current 18-day study shows that the group’s social bias and racism are more likely to be more than just a high percentage of U.S. citizens. That’s because 10,000 people were exposed to torsion-free absences this year.
Many of the victims are the same age groups. The most recent study in U.S. history suggests that the group may be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be the most likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be more likely to be

```

