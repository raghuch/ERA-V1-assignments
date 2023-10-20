This assignment is to perform textual inversion using pre-trained stable diffusion concepts from the huggingface page. I have taken the same functions as defined in the stable-diffusion-deepdive.ipynb notebook, and instead of the blue loss, I have added L1 loss of the generated with respect to a dummy QR code image[qr_code1.png]. The function looks something like:

```
def qr_loss(images, qr_img):

    qr_img = qr_img.unsqueeze(0).to(device)
    error = F.l1_loss(images, qr_img, reduction='mean')
    
    return error
```

where qr_img is the QR code. I have selected 5 styles: "birb", "indian_watercolor", "herge", "midjourney", "marc_allante", each with a different seed (set in the notebook) and then used the same prompt "A picture of James Bond wearing a cowboy hat in the style of puppy" where 'puppy' is the word whose embeddings will be replaced with each of the styles. The idea was to force a QRcode-like image (seen in controlnet), but I couldn't get controlnet-like images. Instead, I have images that are forced to have more shades of black and white, which are seen in the last cell of the notebook[a20.ipynb]