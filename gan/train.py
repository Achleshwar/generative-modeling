from glob import glob
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset

class Rescale():
    def __init__(self):
        pass
    def __call__(self,image):
        image -= 0.5
        image *= 2

        return image

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    # NOTE: don't do anything fancy for 2, hint: the input image is between 0 and 1.
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        Rescale(),
    ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K iterations.
    # The learning rate for the generator should be decayed to 0 over 100K iterations.

    optim_discriminator = optim.Adam(disc.parameters(), lr=0.0002, betas=(0, 0.9))
    scheduler_discriminator = CosineAnnealingLR(optim_discriminator, T_max=500e+3, eta_min=0)
    optim_generator = optim.Adam(gen.parameters(), lr=0.0002, betas=(0,0.9))
    scheduler_generator = CosineAnnealingLR(optim_generator, T_max=100e+3, eta_min=0)
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
    amp_enabled=True,
):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True # speed up training
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="../datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)

    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    fids_list = []
    iters_list = []
    pbar = tqdm(total = num_iterations)
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()
                ############################ UPDATE DISCRIMINATOR ######################################
                # TODO 1.2: compute generator, discriminator and interpolated outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                generator_output = gen(train_batch.shape[0])
                discrim_real = disc(train_batch).reshape(-1)
                discrim_fake = disc(generator_output).reshape(-1)
                # print(discrim_real)
                # print(discrim_fake)
                # discriminator_loss = disc_loss_fn(discrim_real, discrim_fake)

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                epsilon = torch.rand((train_batch.shape[0], 1, 1, 1)).to(device)
                # # print(epsilon.shape)
                # # print(train_batch.shape)
                # # print(generator_output.shape)
                interp = epsilon * train_batch + (1-epsilon) * generator_output
                discrim_interp = disc(interp).reshape(-1)
                discriminator_loss = disc_loss_fn(discrim_real, discrim_fake, discrim_interp, interp, lamb)
            
            
            # discriminator_loss = disc_loss_fn(discrim_real, discrim_fake, discrim_interp, interp, lamb)

            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward(retain_graph=True)
            scaler.step(optim_discriminator)
            # scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # TODO 1.2: compute generator and discriminator output on generated data.
                    generator_output = gen(train_batch.shape[0])
                    discrim_fake = disc(generator_output)
                    generator_loss = gen_loss_fn(discrim_fake)
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()
            
            scheduler_discriminator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        generated_samples = gen(100)
                        generated_samples = torch.clamp(generated_samples, 0.0, 1.0)
                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    if os.environ.get('PYTORCH_JIT', 1):
                        torch.jit.save(torch.jit.script(gen), prefix + "/generator.pt")
                        torch.jit.save(torch.jit.script(disc), prefix + "/discriminator.pt")
                    else:
                        torch.save(gen, prefix + "/generator.pt")
                        torch.save(disc, prefix + "/discriminator.pt")
                    fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
            pbar.update(1)
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
