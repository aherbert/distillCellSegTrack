from data_utils import get_training_and_validation_loaders
from train_utils import train_model, trainEpoch, KD_loss, CPnet
import torch
import numpy as np

def find_seed(n_base, epochs, train_loader, validation_loader, num_iter, device='cuda',progress=False):

    best_val_map_loss = 1000
    best_seed = -1

    min_number = 1
    max_number = 1000000
    list_length = num_iter
    np.random.seed(42)
    random_unique_list = np.random.choice(np.arange(min_number, max_number + 1), size=list_length, replace=False)

    for i in range(num_iter):
        seed = random_unique_list[i]
        torch.manual_seed(seed)
        student_model = CPnet(nbase=n_base, nout=3, sz=3,
                    residual_on=True, style_on=True,
                    concatenation=False, mkldnn=False)

        loss_fn = KD_loss(alpha=2, beta=1)
        optimiser = torch.optim.Adam(student_model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)

        if device is not None:
            student_model = student_model.to(device)

        for epoch in range(epochs):
            student_model, train_y_32_loss, train_map_loss, train_IoU, val_y_32_loss, val_map_loss, val_IoU = \
                trainEpoch(student_model, train_loader, validation_loader, loss_fn, optimiser,
                           scheduler=scheduler, epoch_num=epoch, device=device, progress=False)

        if val_map_loss < best_val_map_loss:
            best_val_map_loss = val_map_loss
            best_seed = seed

        if progress == True and i % 10 == 0:
            print('Tested', str(i+1), 'models.', 'Current best seed: ', seed, '.', 'Current best val_map_loss:',best_val_map_loss,'.')

    return best_seed

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == '__main__':

    #force_cudnn_initialization()

    #This example here is doing the following:
        #Getting the training and validation loaders
        #Getting 40 random seeds
        #Training 40 models (num_iter) iteratively using those random seeds for one epoch each
        #Returning the seed with the lowest val_map_loss

    import os
    base = os.path.dirname(os.path.abspath(__file__));

    cellpose_model_directory = os.path.join(base, "cellpose_models", "Nuclei_Hoechst")
    image_folder = os.path.join(base, "saved_cell_images_1237")

    train_loader, validation_loader = get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel = 0, augment = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    seed = find_seed(n_base=[1,32], epochs=1, train_loader=train_loader, validation_loader=validation_loader, num_iter=40, device=device, progress=True)

    #student_model = train_model([1,32],100,'student_models/resnet_testing',train_loader, validation_loader, device='cuda',progress=True,seed=23944)
