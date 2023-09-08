This is the solution to Assignment-16, basic transformer architecture following the paper. We use it for English to French machine translation using the opus books dataset (from huggingface).
To Run the English - French  MT training, go to train.ipynb and just run the cells.

The current assignment shows the training process with an aim is to get a train loss less than 1.8 (which was achieved in 23 epochs -- I have run more epochs to show the loss stayed consistently lower than 1.8).
Changes made from the original classroom transformer basic training:

 * Add torch.cuda.amp.autocast for mixed precision training -- brought down training time per epoch to 9 mins.
 * Add dynamic padding (other approaches took longer to iterate on, so I justed used the same approach/code shown in class) -- this reduced training time per epoch to 3 mins.
 * Add LR scheduler -- again, no best LR, just set max LR = 1e-3 which is reached in 30% of max_epochs, div factor = 100, strategy=linear -- useful for getting faster convergence than without scheduler.

 Please check the train.ipynb for the training logs and results.
