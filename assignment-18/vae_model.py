import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import pytorch as pl

def label_to_onehot(labels, num_classes):
    labels_onehot = torch.zeros(labels.size()[0], num_classes)
    labels_onehot.scatter_(1, labels.view(-1, 1), 1)

    return labels_onehot

class VAE(pl.LightningModule):
    def __init__(self, in_channels, num_classes, latent_dim, dropout_val=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.enc_conv1 = nn.Sequential(            
            nn.Conv2d(in_channels + self.num_classes, 32, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_val)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout_val),
            nn.MaxPool2d(2, 2)
        )

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout_val)
        )

        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout_val),
            nn.MaxPool2d(2, 2)
        )
        
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout2d(dropout_val)
        )

        self.fc_mu = nn.Linear(512* 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(512* 1 * 1, latent_dim)

        self.dec_linear1 = nn.Linear(latent_dim, 512 * 1 * 1)
        self.dec_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)
        self.dec_conv5 = nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=0, output_padding=0, bias=False)


    def reparametrize(self, mu, log_var, eps=1.0):
        eps = torch.randn(mu.size(0), mu.size(1)).to(mu.get_device())
        z = mu + eps * torch.exp(log_var/2)

        # mean = self.fc_mu(x)
        # logvar = self.fc_logvar(x)
        # std = torch.exp(logvar/2)
        # q = torch.distributions.Normal(mean, std)
        # z = q.rsample()
        return z
    

    def encode_fn(self, x, targets):

        onehot_targets = label_to_onehot(targets, self.num_classes)
        onehot_targets = onehot_targets.view(-1, self.num_classes, 1, 1)
        ones = torch.ones(x.size()[0], self.num_classes, x.size()[2], x.size()[3])
        ones = ones * onehot_targets

        x = torch.cat((x, ones), dim=1)  



        #x = self.encoder(x)
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = self.enc_conv4(x)
        x = self.enc_conv5(x)
        x = x.view(x.size(0), -1)
        z_mean = self.fc_mu(x)
        z_log_var = self.fc_logvar(x)
        encoded_latent = self.reparametrize(z_mean, z_log_var)

        return z_mean, z_log_var, encoded_latent
    
    
    def decode_fn(self, x, encoded, target):

        pass

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        encoded = self.reparametrize(mu, log_var)

        reconstructed = self.decode(encoded)
        return encoded, mu, log_var, reconstructed

    

