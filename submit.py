import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader,random_split
from tqdm import tqdm
from data_load.Load import *
from models.vgg import *

def predict(model, device, test_loader):
    model.eval()
    submit = []
    for batch_idx, dct in enumerate(test_loader):
        images = dct['image']
        images = images.to(device)
        images = images.float()
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        pred = pred.squeeze(dim=1)
        pred = pred.cpu().numpy()
        submit.append(pred)
    submit = np.array(submit)
    return submit

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #data_loader
    test_dataset = mnist_test('./data/test.csv')
    test_loader = DataLoader(test_dataset,batch_size = 100,shuffle=False,num_workers=3)

    model_path = './MNIST.pth'
    model = My_VGG(10)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # model = nn.DataParallel(model)
    nparr = predict(model, device, test_loader)
    nparr = nparr.flatten()
    no_of_pic = range(1,len(nparr)+1)
    no_of_pic = np.array(no_of_pic)
    submit = np.stack((no_of_pic,nparr))
    submit = submit.transpose(1,0)
    np.savetxt("submission.csv", submit, delimiter=",", header="ImageId,Label", fmt="%i",comments='')

if __name__ == '__main__':
    main()