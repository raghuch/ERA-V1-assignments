This dir in this repo is a work to run the UNet model with 4 different configs:

    * Maxpool + Transpose conv + BCE loss

    * Maxpool + Transpose conv + Dice loss

    * StrConv + Transpose conv + BCE loss

    * StrConv + Upsample + Diceloss

The model can be run by just running the cells in run_unet.ipynb jupyter notebook in the same dir. It can be observed that BCE loss converges quite fast (val loss ~ 0.04 after 20 epochs) with or without maxpool and nn.Upsample. But using dice loss results in very slow convergence (val loss ~ 0.98 after 20 epochs) with or without maxpool and nn.Upsample. nn.Upsample or strided conv have almost used same time (probably 20 epochs is too less to observe a difference)
