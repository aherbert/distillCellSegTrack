import os;
from data_utils import get_training_and_validation_loaders
from train_utils import train_model
from method_seeding import find_seed

if __name__ == '__main__':

    base = os.path.dirname(os.path.abspath(__file__));

    cellpose_model_directory = os.path.join(base, "cellpose_models", "Nuclei_Hoechst")
    image_folder = os.path.join(base, "saved_cell_images_1237")
    device = 'cuda'

    #channel 0: nuclei
    #channel 1: cell
    #no channel: both cell and nuclei
    train_loader, validation_loader = get_training_and_validation_loaders(cellpose_model_directory, image_folder, channel = 0, augment = False)

    n_base = [1,32] #model's dimensions
    num_iter = 10 #number of seeds to test for seed searching
    epochs_per_model = 1 #number of epochs to train the models that we test with different seeds
    seed = find_seed(n_base=n_base, epochs=1, train_loader=train_loader, validation_loader=validation_loader, num_iter=num_iter, device=device,progress=True)

    student_model = train_model([1,32],100,'student_models/resnet_testing',train_loader, validation_loader, device=device,progress=True,seed=seed)
