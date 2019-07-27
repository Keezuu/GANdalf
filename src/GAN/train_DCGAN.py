from src.resources.gan_utilities import *
# Hyper-parameters
from torchvision.transforms import transforms

from src.GAN.Discriminator import Discriminator
from src.GAN.Generator import Generator
from src.resources.utilities import *

# Get current date for naming folders
date = datetime.datetime.now().strftime("%m%d%H%M%S")

# Sets up the training dirs - createes them if they dont exist
set_up_training_dirs(date)

# Image processing
# Normalize the images to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])

])

# Get dataloader
data_loader = get_data(transform)

# Get GPU
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# For better readability
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Create discriminator and generator and force them to use GPU
D = Discriminator(img_shape=(28, 28), n_classes=10).cuda()

# Create generator
G = Generator(n_classes=10, latent_dim=cnst.GAN_LATENT_SIZE).cuda()

# Apply the weights init with value from a Normal distribution with mean=0, stdev=0.02.
# As stated in DCGAN paper
D.apply(weights_init)
G.apply(weights_init)

# Visualize networks
print(G)
print(D)

# Create MSE loss function
mse_loss = nn.MSELoss().cuda()

# Create optimizers as specified in DCGAN paper
G_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Statistics to be saved
d_losses = np.zeros(cnst.GAN_NUM_EPOCHS)
g_losses = np.zeros(cnst.GAN_NUM_EPOCHS)
real_scores = np.zeros(cnst.GAN_NUM_EPOCHS)
fake_scores = np.zeros(cnst.GAN_NUM_EPOCHS)

# Training
batches_done = 0
total_step = len(data_loader)
for epoch in range(cnst.GAN_NUM_EPOCHS):
    for i, (imgs, labels) in enumerate(data_loader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = FloatTensor(batch_size, 1).fill_(1.0).cuda()
        fake = FloatTensor(batch_size, 1).fill_(0.0).cuda()

        # Configure input
        real_imgs = imgs.cuda()
        real_labels = labels.cuda()

        # Train DCGAN on one batch
        g_loss, d_loss, real_score, fake_score = batch_train_gan(G, D, G_opt, D_opt, mse_loss, batch_size, real_imgs,
                                                                 real_labels, valid, fake, device)

        # Update statistics
        d_losses[epoch] = d_losses[epoch] * (i / (i + 1.)) + d_loss.data * (1. / (i + 1.))
        g_losses[epoch] = g_losses[epoch] * (i / (i + 1.)) + g_loss.data * (1. / (i + 1.))
        real_scores[epoch] = real_scores[epoch] * (i / (i + 1.)) + real_score.mean().data * (1. / (i + 1.))
        fake_scores[epoch] = fake_scores[epoch] * (i / (i + 1.)) + fake_score.mean().data * (1. / (i + 1.))

        # Print progress
        if i % 4 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch, cnst.GAN_NUM_EPOCHS, i + 1, total_step, d_loss.data, g_loss.data,
                          real_score.mean().data, fake_score.mean().data))

        batches_done = epoch * len(data_loader) + i
        plt.clf()

    # Show samples for debug purpose
    show_samples(G, epoch, device)

    # Save real images
    if (epoch + 1) == 1:
        images = imgs.view(imgs.size(0), 1, 28, 28)
        save_image(denorm(imgs.data), os.path.join(cnst.GAN_SAMPLES_DIR, date, 'real_images.png'))
        # Save text info about this train run
        save_info(path=os.path.join(cnst.GAN_SAMPLES_DIR, date, 'info.txt'), epochs=cnst.GAN_NUM_EPOCHS,
                batch=batch_size, dis_feature_maps=cnst.GAN_DIS_FEATURE_MAPS, gen_feature_maps=cnst.GAN_GEN_FEATURE_MAPS)

    # Save sampled images
    #if epoch % 5 == 0:
    sample_image(G, n_row=10, name=str(epoch).zfill(len(str(cnst.GAN_NUM_EPOCHS))),
                 path=os.path.join(cnst.GAN_SAMPLES_DIR, date))

    # Save and plot Statistics
    save_statistics(d_losses, g_losses, fake_scores, real_scores, os.path.join(cnst.GAN_SAVE_DIR, date))

    # Save model at checkpoints
    if (epoch + 1) % 5 == 0:
        torch.save(G.state_dict(), os.path.join(cnst.GAN_MODEL_DIR, date, 'G--{}.ckpt'.format(epoch + 1)))
        torch.save(D.state_dict(), os.path.join(cnst.GAN_MODEL_DIR, date, 'D--{}.ckpt'.format(epoch + 1)))

# Save the model checkpoints
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')

# generate gif visualizing progress of training
filenames = os.listdir(os.path.join(cnst.GAN_SAMPLES_DIR, date, "img"))
generate_gif(filenames, save_path=os.path.join(cnst.GAN_SAVE_DIR, date),
             read_path=os.path.join(cnst.GAN_SAMPLES_DIR, date))
