import os
import pickle
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import hmax

# initialize the model with the universal patch set
model = hmax.HMAX(os.path.join(os.getcwd(), "universal_patch_set.mat"))
device = torch.device("cuda:0")
print("Running model on", device)

# chunk list for 4 sec (complete)
chunk_list = [0, 253, 252, 242, 275, 249, 243, 306, 178]

class single_dataset():
    """
    taken from: https://github.com/wmvanvliet/pytorch_hmax/issues/1 
    """
    def __init__(self, img_dir, img_labels=None, transform=None):
        self.img_labels = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)

        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_imgs(folder_path):
    """
    function to load all imgs for a corresponding 4sec 
    scene. Since: 25 fps, should be 100 imgs in total.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(size=(250, 250)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        ])
    return single_dataset(
        img_dir=folder_path,
        transform=transform
        )

for run in list(range(1, 9)):
    hmax_out = os.path.join(os.getcwd(), "hmax_output", f"run-{run}")

    for chunk in list(range(1, chunk_list[run]+1)):
        print(f"starting run-{run} and chunk-{chunk} ...")
        in_folder = os.path.join(os.getcwd(), "ssim_movie-frames", f"raw_run-{run}", f"chunk-{chunk}")

        # initialize dataset
        input_imgs = load_imgs(folder_path=in_folder)
        dataset = DataLoader(input_imgs)

        model = model.to(device)
        for X, y in dataset:
            s1, c1, s2, c2 = model.get_all_layers(X.to(device))

        print(f"saving output of all layers to: run-{run}_chunk-{chunk}_activations.pkl")
        os.makedirs(hmax_out, exist_ok=True)
        with open(os.path.join(hmax_out, f"run-{run}_chunk-{chunk}_c1-activations.pkl"), "wb") as f:
            pickle.dump(dict(c1=c1), f)

print("[all done]")