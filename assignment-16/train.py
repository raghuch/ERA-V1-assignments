import warnings
from model import build_transformer
from dataset import BiLingualDataset, causal_mask
from config import get_config, get_weights_file_path
import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import LambdaLR

import os
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard.writer import SummaryWriter
from torchsummary import summary



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    #Precompute encoder output and use it for every step
    encoder_output = model.encode(source, source_mask)
    #Initialize the decoder with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        #build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        #get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, 
                                   torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        #topv, topi = out.topk(1)
        
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])

    return model

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):

    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:

        with os.popen('ssty size','r') as console:
          _, console_width = console.read().split()
          console_width = int(console_width)

    except:
        console_width =80

    with torch.no_grad():
        for batch in validation_ds:
          count += 1
          encoder_input = batch["encoder_input"].to(device)
          encoder_mask = batch["encoder_mask"].to(device)
          assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

          model_out = greedy_decode(model,encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

          source_text = batch["src_text"][0]
          target_text = batch["tgt_text"][0]

          model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

          source_texts.append(source_text)
          expected.append(target_text)
          predicted.append(model_out_text)

          print_msg('-'*console_width)
          print_msg(f"{f'SOURCE: ':>12}{source_text}")
          print_msg(f"{f'TARGET: ':>12}{target_text}")
          print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

          if count == num_examples:
            print_msg('-'*console_width)
            break

        if writer:
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer,global_step)
            writer.flush()

            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer,global_step)
            writer.flush()

            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu,global_step)
            writer.flush()

# run tokenizer
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds,lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def filter_ds(ds_raw, config,token_limit=150, token_len_diff=10):
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    tokenizer_src = Tokenizer.from_file(f"{config['tokenizer_file'].format(src_lang)}")
    tokenizer_tgt = Tokenizer.from_file(f"{config['tokenizer_file'].format(tgt_lang)}")

    len_params = []
    for i in range(len(ds_raw)):

        len_params.append({src_lang: len(tokenizer_src.encode(ds_raw[i]['translation'][src_lang])),
                          tgt_lang: len(tokenizer_tgt.encode(ds_raw[i]['translation'][tgt_lang])) })

    ds_raw = ds_raw.add_column(name='token_length', column=len_params )
    
    ds_raw = ds_raw.filter(lambda x: x['token_length'][src_lang] < token_limit)
    ds_raw = ds_raw.filter(lambda x:abs(x['token_length'][tgt_lang] - x['token_length'][src_lang]) <= token_len_diff)

    return ds_raw

def get_ds(config):

    ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split='train' )

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    ds_filtered = filter_ds(ds_raw, config, token_limit=150, token_len_diff=10)

    train_ds_size = int(0.9 * len(ds_filtered))
    val_ds_size = len(ds_filtered) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_filtered, [train_ds_size, val_ds_size])

    train_ds = BiLingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BiLingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_filtered:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'],shuffle=True,num_workers = 16, collate_fn=collate_fn_dynamic_padding)
    val_dataloader = DataLoader(val_ds, batch_size = 1,shuffle=True,num_workers = 16)

    return train_dataloader,val_dataloader, tokenizer_src, tokenizer_tgt

def collate_fn_dynamic_padding(batch):
    encoder_input_max = max(x["encoder_str_length"] for x in batch)
    decoder_input_max = max(x["decoder_str_length"] for x in batch)

    encoder_inputs = []
    decoder_inputs = []
    encoder_mask = []
    decoder_mask = []
    label = []
    src_text = []
    tgt_text = []

    for b in batch:
        encoder_inputs.append(b["encoder_input"][:encoder_input_max])
        decoder_inputs.append(b["decoder_input"][:decoder_input_max])
        encoder_mask.append(b["encoder_mask"][0, 0, :encoder_input_max].unsqueeze(0).unsqueeze(0).unsqueeze(0))
        decoder_mask.append(b["decoder_mask"][0, :decoder_input_max, :decoder_input_max].unsqueeze(0).unsqueeze(0))
        label.append(b["label"][:decoder_input_max])
        src_text.append(b["src_text"])
        tgt_text.append(b["tgt_text"])

    return {
        "encoder_input": torch.vstack(encoder_inputs),
        "decoder_input": torch.vstack(decoder_inputs),
        "encoder_mask": torch.vstack(encoder_mask),
        "decoder_mask": torch.vstack(decoder_mask),
        "label": torch.vstack(label),
        "src_text": src_text,
        "tgt_text": tgt_text
    }

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using device:",device)

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    lr = [0.0]

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-6)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print('preloaded')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['MAX_LR'], steps_per_epoch=len(train_dataloader), epochs=config['num_epochs'],
                                                    pct_start=int(0.3*config['num_epochs'])/config['num_epochs'] if config['num_epochs']!=1 else 0.5,
                                                    div_factor=100, three_phase=False, final_div_factor=100, anneal_strategy='linear')

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc =f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:

          encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
          decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
          encoder_mask  = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
          decoder_mask  = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

          #Run tensors thru encoder, decoder  and projection layer
          with torch.autocast(device_type="cuda", dtype=torch.float16):
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask,decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

          batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

          writer.add_scalar('train loss',loss.item(), global_step)
          writer.flush()

          #loss.backward()
          scaler.scale(loss).backward()

          #optimizer.step()
          scale = scaler.get_scale()
          scaler.step(optimizer)
          scaler.update()
          skip_lr_scheduler = (scale > scaler.get_scale())
          if not skip_lr_scheduler:
              scheduler.step()
          lr.append(scheduler.get_last_lr())
          optimizer.zero_grad(set_to_none=True)

          global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,lambda msg: batch_iterator.write(msg),   global_step, writer)

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
         }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)



