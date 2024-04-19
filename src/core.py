from datetime import datetime
import os
from src.crop import CropAroundBoundingBox
from src.to_device import ToDevice
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from src.constants import TRAIN_MEAN, TRAIN_STD
from src.dog_data_set import DogPoseDataSet
from src.heatmaps import HeatMaps
from src.normalize import CROPPED_TRAIN_DATASET_MEAN, CROPPED_TRAIN_DATASET_STD, Normalize
from src.rescale import Rescale
from src.to_dtype import ToDtype
from src.to_feature_label import ToImageAndHeatMaps
from src.augment import Augment
from torch.utils.data import DataLoader

TorchDevice = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

ImageSize = 96

# This is the core transformer that we will use for all data sets 
Transforms = transforms.Compose([
    CropAroundBoundingBox(),
    # Augment({"Flip": True, "Rotate": True}),
    Rescale(ImageSize),
    # We want to cahnge the image to a float tensor that is scaled between [0, 1]
    ToDtype(dtype=torch.float32, scale=True),
    Normalize(mean=CROPPED_TRAIN_DATASET_MEAN, std=CROPPED_TRAIN_DATASET_STD),
    HeatMaps(std_dev=4),
    ToImageAndHeatMaps(),
])

# Augmentation is skipped for testing and validation sets
TestValidationTransforms = transforms.Compose([
    CropAroundBoundingBox(),
    Rescale(ImageSize),
    # We want to cahnge the image to a float tensor that is scaled between [0, 1]
    ToDtype(dtype=torch.float32, scale=True),
    Normalize(mean=CROPPED_TRAIN_DATASET_MEAN, std=CROPPED_TRAIN_DATASET_STD),
    HeatMaps(std_dev=4),
    ToImageAndHeatMaps(),
])

# This is the transformers for just the image without normalization  
TransformsBasic = transforms.Compose([
    CropAroundBoundingBox(),
    # Augment({"Flip": True, "Rotate": True}),
    Rescale(ImageSize),
    HeatMaps(std_dev=4),
])

TransformsBasicNoAugmentation = transforms.Compose([
    CropAroundBoundingBox(),
    Rescale(ImageSize),
    HeatMaps(std_dev=4),
])

TestDataSetNoNormalization = DogPoseDataSet(
    images_dir = "../data/Images/",
    np_split_file="../data/annotations/test_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/test_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=TransformsBasicNoAugmentation)

TrainDataSetNoNormalization = DogPoseDataSet(
    images_dir = "../data/Images/", 
    np_split_file="../data/annotations/train_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/train_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=TransformsBasic)

ValidationDataSetNoNormalization = DogPoseDataSet(
    images_dir = "../data/Images/",
    np_split_file="../data/annotations/val_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/val_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=TransformsBasicNoAugmentation)


TrainDataSet = DogPoseDataSet(
    images_dir = "../data/Images/", 
    np_split_file="../data/annotations/train_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/train_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=Transforms)

ValidationDataSet = DogPoseDataSet(
    images_dir = "../data/Images/",
    np_split_file="../data/annotations/val_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/val_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=TestValidationTransforms)

TestDataSet = DogPoseDataSet(
    images_dir = "../data/Images/",
    np_split_file="../data/annotations/test_stanford_StanfordExtra_v12_new_split.npy",
    annotations_json_file="../data/annotations/StanfordExtra_v12.json",
    np_skip_file="../data/annotations/test_skip_stanford_StanfordExtra_v12_new_split.npy",
    transform=TestValidationTransforms)

TrainDataLoader = DataLoader(
    TrainDataSet,
    batch_size=8,
    shuffle=True, # Shuffling is important for training
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

# Only used for testing
TrainDataLoaderNoShuffle = DataLoader(
    TrainDataSet,
    batch_size=8,
    shuffle=False, # Shuffling is important for training
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

ValidationDataLoader = DataLoader(
    ValidationDataSet,
    batch_size=8,
    shuffle=False, # No shuffling needed for validation set because we want to compare loss at each epoch
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

TestDataLoader = DataLoader(
    TestDataSet,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

TestDataSetNoNormalizationLoader = DataLoader(
    TestDataSetNoNormalization,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

TrainDataSetNoNormalizationLoader = DataLoader(
    TrainDataSetNoNormalization,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

ValidationDataSetNoNormalizationLoader = DataLoader(
    ValidationDataSetNoNormalization,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator(device=TorchDevice),
)

# Optimizers specified in the torch.optim package
# RMSProp is good for image classification
def get_optimizer(model):
    return torch.optim.Adam(model.parameters())

# Loss Function MSE is good for regression
def get_loss_fn():
    return torch.nn.MSELoss()


def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(TrainDataLoader):
        # print(f"EPOCH {epoch_index + 1}: processing batch {i}")
        # Every data instance is an input + label pair
        inputs, labels = data

        # Required to change the device to cuda
        inputs, labels = inputs.to(TorchDevice), labels.to(TorchDevice)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report on every 50th batch 
        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(TrainDataLoader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def train_model(model, optimizer, loss_fn, epochs, run_dir, model_dir):
    os.makedirs(run_dir, exist_ok = True) 
    os.makedirs(model_dir, exist_ok = True) 
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer_dir = os.path.join(run_dir, f'model_{timestamp}')

    writer = SummaryWriter(writer_dir)
    epoch_number = 0

    best_vloss = float('inf')

    for epoch in range(epochs):
        print(f"EPOCH {epoch_number + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(ValidationDataLoader):
                # print(f"EPOCH {epoch_number + 1}: processing validation data batch {i}")
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(TorchDevice), vlabels.to(TorchDevice)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(model_dir, f'model_{timestamp}_{epoch_number}')
            torch.save(model.state_dict(), model_path)

        epoch_number += 1