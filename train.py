import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader,random_split
from tqdm import tqdm
from data_load.Load import *
from models.vgg import *
from torch.autograd import Variable

def train(args,model,device,train_loader,optim,epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx,dct in enumerate(train_loader):
        images_in_batch = dct['image']
        labels_in_batch = dct['label']
        images_in_batch = images_in_batch.to(device)
        labels_in_batch = labels_in_batch.to(device)
        labels_in_batch = labels_in_batch.view(-1)
        optim.zero_grad()

        images_in_batch = images_in_batch.float()
        outputs = model(images_in_batch)
        loss = criterion(outputs,labels_in_batch)
        loss.backward()
        optim.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_in_batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, dct in enumerate(test_loader):
            images = dct['image']
            labels = dct['label']
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.view(-1)

            images = images.float()
            output = model(images)
            test_loss = criterion(output, labels) # sum up batch loss
            total_loss+=test_loss
            count+=1
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
        if batch_idx % args.log_interval == 0:
            print('test loss: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(images), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), test_loss.item()))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss/count, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    #data_loader
    all_data = mnist_all('./data/train.csv',transform = transforms.Compose([ToTensor()]))

    train_size = int(0.8 * len(all_data))
    test_size = len(all_data) - train_size
    train_dataset, test_dataset = random_split(all_data, [train_size, test_size])

    train_loader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle=True,num_workers=3)
    test_loader = DataLoader(test_dataset,batch_size = args.test_batch_size,shuffle=True,num_workers=3)


    model = My_VGG(10).to(device)
    # model = nn.DataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        print("total epoch",args.epochs)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"MNIST.pth")
        
if __name__ == '__main__':
    main()