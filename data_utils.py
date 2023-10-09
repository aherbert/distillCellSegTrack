import os
import glob
import numpy as np
import torch
import logging
from cellpose import transforms
from cellpose_ext import CPnetX
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def pred(x, network):
    """ convert imgs to torch and run network model and return numpy """
    X = x
    with torch.no_grad():
        #channel_32_output, final_output = network(X, training_data=True)
        final_output, _, channel_32_output = network(X)
        return channel_32_output, final_output

def make_tiles_and_reshape(img, bsize, augment, tile_overlap):
    tiles, _, _, _, _ = transforms.make_tiles(img, bsize=bsize, augment=augment, tile_overlap=tile_overlap)
    ny, nx, nchan, ly, lx = tiles.shape
    return np.reshape(tiles, (ny*nx, nchan, ly, lx))

def run_tiled(imgi, network, augment=False, bsize=224, tile_overlap=0.1, ):
    tiles = []
    channel_32_outputs = []
    final_outputs = []

    # make image into tiles
    IMG = make_tiles_and_reshape(imgi, bsize, augment, tile_overlap)

    # process in batches
    batch_size = 8
    niter = int(np.ceil(IMG.shape[0] / batch_size))

    for k in range(niter):
        irange = slice(batch_size * k, min(IMG.shape[0], batch_size * (k + 1)))
        input_img = torch.from_numpy(IMG[irange])
        tiles.append(input_img)

        # predict on the input image
        channel_32_output, final_output = pred(input_img, network)
        channel_32_outputs.append(channel_32_output)
        final_outputs.append(final_output)

    return tiles, channel_32_outputs, final_outputs

def run_net(imgs, network, augment=False, tile_overlap=0.1, bsize=224,
                 # Unused
                 return_conv=False,return_training_data=False):

        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2,0,1))

        # pad image for net so Ly and Lx are divisible by 4
        imgs, _, _ = transforms.pad_image_ND(imgs)

        tiles, channel_32_outputs, final_outputs = run_tiled(imgs, network=network, augment=augment, bsize=bsize,
                                    tile_overlap=tile_overlap)
        return tiles, channel_32_outputs, final_outputs

def run_cp(x, network, normalize=True, invert=False, augment=False): #, tile=True):

    iterator = range(x.shape[0])

    tiled_images_input = []
    intermdiate_outputs = []
    flows_and_cellprob_output = []
    for i in iterator:
        img = np.asarray(x[i])
        if normalize or invert:
            img = transforms.normalize_img(img, invert=invert)

        tiles, channel_32_outputs, final_outputs = run_net(img, network=network, augment=augment, tile_overlap=0.1, bsize=224,
                return_conv=False, return_training_data=True)
        tiled_images_input.append(tiles)
        intermdiate_outputs.append(channel_32_outputs)
        flows_and_cellprob_output.append(final_outputs)
    return tiled_images_input, intermdiate_outputs, flows_and_cellprob_output

def get_clean_data(data_input):
    clean_data = []

    for i in range(len(data_input[0])):
        # Remove the channel at dimension 1
        #tiles = np.delete(data_input[0][i], 1, 1)
        tiles = np.squeeze(data_input[0][i])

        for tile in tiles:
            clean_data.append(tile)

    return torch.stack(clean_data)

def get_clean_train_data(tiled_images_input, intermediate_outputs, flows_and_cellprob_output):
    tiled_images_input_train_clean = get_clean_data(tiled_images_input)
    tiled_intermediate_outputs_train_clean = get_clean_data(intermediate_outputs)
    tiled_flows_and_cellprob_output_train_clean = get_clean_data(flows_and_cellprob_output)

    # Data is in format (n,X,Y,C).
    # Change to (n,C,X,Y). The -1 allows this dimension to be inferred.
    # This call will fail if the default tile size of 224 is changed.
    tiled_images_input_train_clean = tiled_images_input_train_clean.reshape(-1, 2, 224, 224)

    return tiled_images_input_train_clean, tiled_intermediate_outputs_train_clean, tiled_flows_and_cellprob_output_train_clean

def get_data_cp_clean(unet,combined_images):
    tiled_images_final = []
    intermediate_outputs_final = []
    flows_and_cellprob_output_final = []

    for i in range(len(combined_images)):
        logging.info(f'Processing {i+1}/{len(combined_images)}')

        x = combined_images[i]

        if not isinstance(x, np.ndarray):
            x = np.array(x)

        while x.ndim < 4:
            x = x[np.newaxis,...]

        # Change view from (n,C,X,Y) to (n,X,Y,C)
        x = x.transpose((0,2,3,1))

        tiled_images_input, intermdiate_outputs, flows_and_cellprob_output = run_cp(x,unet)
        tiled_images_input_train_clean, tiled_intermediate_outputs_train_clean, tiled_flows_and_cellprob_output_train_clean = get_clean_train_data(tiled_images_input, intermdiate_outputs, flows_and_cellprob_output)
        tiled_images_final.append(tiled_images_input_train_clean)
        intermediate_outputs_final.append(tiled_intermediate_outputs_train_clean)
        flows_and_cellprob_output_final.append(tiled_flows_and_cellprob_output_train_clean)

    tiled_images_final = torch.cat(tiled_images_final, dim=0)
    intermediate_outputs_final = torch.cat(intermediate_outputs_final, dim=0)
    flows_and_cellprob_output_final =  torch.cat(flows_and_cellprob_output_final, dim=0)

    return tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final

def adjust_brightness(image, brightness_factor):
    # Ensure the image has values in the range [0, 1]
    #image = torch.clamp(image, 0.0, 1.0)
    # Apply brightness adjustment
    adjusted_image = image * brightness_factor
    # Clip again to ensure values are still in the range [0, 1]
    # adjusted_image = torch.clamp(adjusted_image, 0.0, 1.0)
    return adjusted_image

def apply_rotation(original_image):
    # Rotate by 90 degrees
    rotated_90 = torch.rot90(original_image, k=1, dims=(1, 2))
    # Rotate by 180 degrees
    rotated_180 = torch.rot90(original_image, k=2, dims=(1, 2))
    # Rotate by 270 degrees
    rotated_270 = torch.rot90(original_image, k=3, dims=(1, 2))
    return rotated_90, rotated_180, rotated_270

def brightness_augmentation(tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final):
    brightness_min = 0.1
    brightness_max = 3

    new_tiled_images_final = []
    new_intermediate_outputs_final = []
    new_flows_and_cellprob_output_final = []

    # Generate a list of random numbers that is the same length as range(tiled_images_final.shape[0])
    np.random.seed(42)
    random_numbers = np.random.uniform(0, 1, tiled_images_final.shape[0])

    # Iterate through the tensors in tiled_images_final
    # Image shape (n,C,X,Y)
    for image_idx in range(tiled_images_final.shape[0]):
        # Get the original image tensor from tiled_images_final
        original_image = tiled_images_final[image_idx]

        # Append the original tensor (without brightness changes) to the list
        new_tiled_images_final.append(original_image)
        new_intermediate_outputs_final.append(intermediate_outputs_final[image_idx])
        new_flows_and_cellprob_output_final.append(flows_and_cellprob_output_final[image_idx])

        # Apply random brightness change with a probability of 0.5
        # Why do this with a 50% probability?
        if random_numbers[image_idx] < 0.5:
            brightness_factor = torch.tensor([torch.FloatTensor(1).uniform_(brightness_min, brightness_max)])
            new_image = adjust_brightness(original_image, brightness_factor)

            # Append the new tensor with brightness changes to the list
            new_tiled_images_final.append(new_image)
            # XXX: Expect the intermediates and probability output to be the same
            new_intermediate_outputs_final.append(intermediate_outputs_final[image_idx])
            new_flows_and_cellprob_output_final.append(flows_and_cellprob_output_final[image_idx])

    # Convert the lists to PyTorch tensors
    tiled_images_final = torch.stack(new_tiled_images_final)
    intermediate_outputs_final = torch.stack(new_intermediate_outputs_final)
    flows_and_cellprob_output_final = torch.stack(new_flows_and_cellprob_output_final)

    return tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final

def rotation_augmentation(tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final):
    new_tiled_images_final = []
    new_intermediate_outputs_final = []
    new_flows_and_cellprob_output_final = []

    # Generate a list of random numbers that is the same length as range(tiled_images_final.shape[0])
    np.random.seed(43)
    random_numbers = np.random.uniform(0, 1, tiled_images_final.shape[0])

    # Iterate through the tensors in tiled_images_final
    # Image shape (n,C,X,Y)
    for image_idx in range(tiled_images_final.shape[0]):
        new_tiled_images_final.append(tiled_images_final[image_idx])
        new_intermediate_outputs_final.append(intermediate_outputs_final[image_idx])
        new_flows_and_cellprob_output_final.append(flows_and_cellprob_output_final[image_idx])

        # Why do this with a 50% probability?
        if random_numbers[image_idx] < 0.5:
            # Append the rotated tensors to the list
            # Q. Is this valid?
            new_tiled_images_final.extend(apply_rotation(tiled_images_final[image_idx]))
            new_intermediate_outputs_final.extend(apply_rotation(intermediate_outputs_final[image_idx]))
            new_flows_and_cellprob_output_final.extend(apply_rotation(flows_and_cellprob_output_final[image_idx]))

    # Convert the lists to PyTorch tensors
    tiled_images_final = torch.stack(new_tiled_images_final)
    intermediate_outputs_final = torch.stack(new_intermediate_outputs_final)
    flows_and_cellprob_output_final = torch.stack(new_flows_and_cellprob_output_final)

    return tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final

def create_dataset(cellpose_model_file, images, channel=None, augment=False, device=None):

    # Only the architecture, easier to extract the data
    cpnet = CPnetX(nbase=[2, 32, 64, 128, 256], nout=3, sz=3, residual_on=True)
    cpnet.load_model(cellpose_model_file, device=(torch.device(device) if device else None))

    combined_images = []
    if np.isscalar(images):
        images = [images]
    for f in images:
        if os.path.isdir(f):
            for filename in glob.glob(os.path.join(f, '*.npy')):
                combined_images.append(np.load(filename))
        elif os.path.isfile(f):
            combined_images.append(np.load(f))
        else:
            raise FileNotFoundError(f)           

    # XXX: This makes assumptions that the input images have 1/2 channels
    if channel != None:
        combined_images = np.array(combined_images)
        # XXX: deletes the specified channel (should it delete the others?)
        # assumes a dual channel image
        combined_images = np.delete(combined_images, channel, 1)
        # XXX: why duplicate the single channel back to 2 channels?
        # if running on 1 channel then the system should handles smaller
        # combined images array. Note the channel is deleted from the
        # train and validation images below
        combined_images = np.repeat(combined_images, 2, axis=1)
    else:
        combined_images = np.array(combined_images)

    logging.info(f'Processing {len(combined_images)} images')
    tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final = get_data_cp_clean(
        cpnet, combined_images)

    if augment == True:
        logging.info('Processing augmentations')
        # TODO:
        # Verify if rotation of the CP outputs is valid. Otherwise we will have
        # to rotate the tiles and run through CPnet.

        # Convert the lists to PyTorch tensors
        tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final = rotation_augmentation(tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final)
        tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final = brightness_augmentation(tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final)

    if channel != None:
        # XXX: Should this remove the other channel(s)?
        #remove the second channel of train_images_tiled and val_images_tiled
        tiled_images_final = np.delete(tiled_images_final, channel, 1)

    return tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final

def get_training_and_validation_loaders(cellpose_model_file, images, channel=None, augment=False, device=None):

    tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final = create_dataset(
        cellpose_model_file, images, channel=channel, augment=augment, device=device)

    logging.info('Splitting into training and validation data')
    train_images_tiled, val_images_tiled, train_upsamples, val_upsamples, train_ys, val_ys = train_test_split(
        tiled_images_final, intermediate_outputs_final, flows_and_cellprob_output_final,
        test_size=0.1, random_state=42)

    train_dataset = ImageDataset(train_images_tiled, train_upsamples, train_ys)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    validation_dataset = ImageDataset(val_images_tiled, val_upsamples, val_ys)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)

    return train_loader, validation_loader

class ImageDataset(Dataset):
    def __init__(self, image, upsample, cellprob):
        self.image = image
        self.upsample = upsample
        self.cellprob = cellprob

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        upsample = self.upsample[idx]
        cellprob = self.cellprob[idx]
        return img, upsample, cellprob
