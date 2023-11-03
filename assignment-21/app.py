import gradio as gr

from bigram_model import BigramLanguageModel
import os
from data_utils import *
import torch

def generate_nanogpt_text():
    model = BigramLanguageModel(vocab_size=65, n_embed=n_embed, block_size=BLOCK_SIZE, num_heads=n_head, n_layers=n_layer)
    ckpt = torch.load(os.path.join("./nano_gpt_ckpts", "ckpt_5k_iters.pt"))
    model.load_state_dict(ckpt['model'])

    char_tokenizer = load_int_char_tokenizer(load_text())

    context = torch.zeros((1, 1), dtype=torch.long)
    generated_text = char_tokenizer.decode(model.generate(context, max_new_tokens=400)[0].tolist())

    return generated_text


#gr.Interface(fn=generate_nanogpt_text, inputs=gr.Button(value="Generate text!"), outputs='text').launch(share=True)


with  gr.Blocks() as demo:
    generate_button = gr.Button("Generate text!")
    output = gr.Textbox(label="Generated text from nano-gpt")
    generate_button.click(fn=generate_nanogpt_text, inputs=None, outputs=output, api_name='nano-gpt text generation sample')

demo.launch(share=True)
