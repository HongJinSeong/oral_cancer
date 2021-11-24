
import skimage.io as iio
from utility.datasetutils import *

from torch.utils.data import DataLoader
from torchvision import  transforms

import matplotlib.pyplot as plt
orals=glob('datasets/oralsamples', '*', True)

tform_train=transforms.Compose([
    transforms.Resize(size=(525,525)),
    transforms.RandomResizedCrop(size=(500,500), scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=2),
    transforms.RandomHorizontalFlip(),           ### 좌우(mirroring)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



ds=oralDataset_test(orals,tform_train)

traindataloader = DataLoader(ds, batch_size=1, shuffle=True)

for i,datas in enumerate(traindataloader):
    names,inputs=datas
    d1 = inputs[0].permute(1, 2, 0).numpy()
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(1, 1, 1)
    plt.imshow(d1)

    plt.show()

