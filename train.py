"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
from torchvision import transforms
import numpy as np
import GAN
from tqdm import tqdm
import matplotlib.pyplot as plt

# N = 100


def train(G, D, trainloader, optimizer_G, optimizer_D, epoch, loss_fn, device, loss_type='standard'):
    G.train()  # set to training mode
    D.train()
    G_losses = []
    D_losses = []
    #
    # BCE_loss = -[yn*log(xn) + (1-yn)*log(1-xn)]
    #
    for x, _ in trainloader:
        real = x.view(-1, 784).to(device)
        # Update the discriminator function
        for i in range(2):
            optimizer_D.zero_grad()
            sample_z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
            # BCE_loss = -[yn*log(xn) + (1-yn)*log(1-xn)]
            # want to maximize Disc_loss = log(D(real_x)) + log(1-D(fake)) , fake = G(z)
            # if we put yn = 1 and xn = D(real_x) : BCE loss = -log(D(real_x))
            disc_real = D(real)
            label_real = torch.ones_like(disc_real) 

             # if we put yn = 0 and xn = D(fake): BCE loss = -log(1-D(fake))
            disc_fake = D(G(sample_z))
            label_fake = torch.zeros_like(disc_fake)

            real_loss = loss_fn(disc_real, label_real)
            fake_loss = loss_fn(disc_fake, label_fake)
            disc_loss = real_loss + fake_loss
            disc_loss.backward()
            optimizer_D.step()


        # Update Generator loss
        optimizer_G.zero_grad()
        sample_z = torch.randn(trainloader.batch_size, G.latent_dim).to(device)
        disc_g_z = D(G(sample_z))
        if loss_type == 'original':
            # minimize E(log(1-D(G(z)))
            # if yn = 0 BCE_loss = -log(1-D(fake)) so we want to minimize -(BCE) == log(1-D(fake))
            targets = torch.zeros_like(disc_g_z) 
            g_loss = -1
        else: #loss_type == 'standard'
            # maximize E(log(D(g(z))))
            # if yn == 1 ==> BCE = -log(D(g(z))) so we want to minimize BCE
            targets = torch.ones_like(disc_g_z) 
            g_loss = 1

        g_loss *= loss_fn(disc_g_z, targets)
        g_loss.backward()
        optimizer_G.step()

        G_losses.append(g_loss.item())
        D_losses.append(disc_loss.item())

    generator_loss = sum(G_losses)/ (len(G_losses))
    discriminator_loss = sum(D_losses)/(len(D_losses))
    print(f'Epoch: {epoch}, Generator loss: {generator_loss}, Discriminator_loss: {discriminator_loss}')
    return generator_loss, discriminator_loss


def sample(G, sample_size, save_file, device, filename, epoch, type_loss):
    G.eval()  # set to inference mode
    with torch.no_grad():
        z = torch.randn(sample_size, G.latent_dim).to(device)
        gen_images = G(z)
        if save_file:
            torchvision.utils.save_image(gen_images.view(-1, 1, 28, 28),
                                     'samples/' + filename +f'loss_{type_loss}_' + 'epoch_%d.png' % epoch)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'mid%d_' % args.latent_dim

    G = GAN.Generator(latent_dim=args.latent_dim,device=device, batch_size=args.batch_size).to(device)
    D = GAN.Discriminator().to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)

    loss_fn = torch.nn.BCELoss()
    loss_type = args.loss_type 
    loss_generator_train = []
    loss_discriminator_train = []
    for epoch in tqdm(range(args.epochs)):
        generator_loss, discriminator_loss = train(G, D, trainloader, optimizer_G, optimizer_D, epoch, loss_fn, device, loss_type=loss_type)
        loss_generator_train.append(generator_loss)
        loss_discriminator_train.append(discriminator_loss)
        if (epoch+1)%10==0:
            sample(G, sample_size=args.sample_size, save_file=True, device=device, filename=filename, epoch=epoch, type_loss=loss_type)

        # loss = test(vae, testloader, filename, epoch, device)
        # loss_test.append(loss)
    
    x = [i for i in range(args.epochs)]
    plt.plot(x, loss_generator_train, label='Generator')
    plt.plot(x,  loss_discriminator_train, label='Discriminator')
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f'Loss_vs_epoch_{args.dataset}_loss_type_{loss_type}.png')
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=100)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)

    parser.add_argument('--latent-dim',
                        help='.',
                        type=int,
                        default=100)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=2e-4)
    parser.add_argument('--loss_type',
                            type=str,
                            default='standard') # original
                            

    args = parser.parse_args()
    main(args)
