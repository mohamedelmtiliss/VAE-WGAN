import torch
import torch.nn as nn

# ================= CONFIGURATION =================
LATENT_DIM = 128   # Size of the compressed feature vector
IMG_CHANNELS = 3   # Red, Green, Blue
IMG_SIZE = 64      # 64x64 pixels


class Encoder(nn.Module):
    """
    Compresses the image into Mean and Log-Variance vectors.
    """

    def __init__(self):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(IMG_CHANNELS, 32, 4, 2, 1),  # -> 32 x 32 x 32
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),  # -> 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # -> 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        # Flatten the output (256 * 4 * 4 = 4096)
        self.fc_mu = nn.Linear(256 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(256 * 4 * 4, LATENT_DIM)

    def forward(self, x):
        features = self.net(x)
        features = features.view(features.size(0), -1)  # Flatten
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class Decoder(nn.Module):
    """
    Reconstructs the image from the Latent Vector (z).
    Also acts as the 'Generator' in the GAN context.
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(LATENT_DIM, 256 * 4 * 4)

        self.net = nn.Sequential(
            # Input: 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # -> 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, IMG_CHANNELS, 4, 2, 1),  # -> 3 x 64 x 64
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)  # Reshape back to image map
        img = self.net(x)
        return img


class Discriminator(nn.Module):
    """
    The Critic. Tells us how 'real' an image looks.
    Crucial: Uses InstanceNorm instead of BatchNorm for WGAN stability.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(IMG_CHANNELS, 32, 4, 2, 1),  # -> 32 x 32 x 32
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),  # -> 64 x 16 x 16
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x 8 x 8
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # -> 256 x 4 x 4
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
        )

        # Output one single number (The Score)
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        features = self.net(x)
        features = features.view(features.size(0), -1)
        score = self.fc(features)
        return score

class VAE_WGAN(nn.Module):
    """
    Wrapper class to hold everything together.
    """
    def __init__(self):
        super(VAE_WGAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

    def reparameterize(self, mu, logvar):
        """
        The 'Magic' of VAEs: Sampling from the distribution
        while keeping gradients flowing.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        mu, logvar = self.encoder(x)
        # 2. Sample Z
        z = self.reparameterize(mu, logvar)
        # 3. Decode (Reconstruct)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# # Sanity Check
# if __name__ == "__main__":
#     # Test with a dummy image
#     dummy_img = torch.randn(5, 3, 64, 64)  # Batch of 5 images
#     model = VAE_WGAN()
#     recon, mu, logvar = model(dummy_img)
#     critic_score = model.discriminator(dummy_img)
#
#     print(f"Input Shape: {dummy_img.shape}")
#     print(f"Recon Shape: {recon.shape}")  # Should be [5, 3, 64, 64]
#     print(f"Critic Score Shape: {critic_score.shape}")  # Should be [5, 1]
#     print("Model architecture is valid!")