import argparse
import torch
import time
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def main():
    args = parse_args()
    
    if args.hidden_layers:
        hidden_layers = args.hidden_layers
    else:
        hidden_layers = 500
        
    if args.model == 'densenet169':
        in_features = 1664
        model = models.densenet169(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    
        classifier_new = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, int(hidden_layers))), 
                                                 ('relu1', nn.ReLU()),
                                                 ('dropout1', nn.Dropout(0.5)),
                                                 ('fc2', nn.Linear(int(hidden_layers), int((hidden_layers/2)))),
                                                 ('relu2', nn.ReLU()),
                                                 ('dropout2', nn.Dropout(0.5)),
                                                 ('fc3', nn.Linear(int((hidden_layers/2)), 102)),
                                                 ('output', nn.LogSoftmax(dim = 1))]))    
    
    elif args.model == 'vgg19':
        in_features = 25088
        model = models.densenet169(pretrained = True)
        for param in model.parameters():
            param.required_grad = False
            
        classifier_new = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, int(hidden_layers))), 
                                                 ('relu1', nn.ReLU()),
                                                 ('dropout1', nn.Dropout(0.5)),
                                                 ('fc2', nn.Linear(int(hidden_layers), int((hidden_layers/2)))),
                                                 ('relu2', nn.ReLU()),
                                                 ('dropout2', nn.Dropout(0.5)),
                                                 ('fc3', nn.Linear(int((hidden_layers/2)), 102)),
                                                 ('output', nn.LogSoftmax(dim = 1))]))
        
    model.classifier = classifier_new
    
    data_transforms, image_datasets, dataloaders = get_data(args.data_path)
    
    if args.gpu:
        model = model.to('cuda')
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.lr)
    
    train_DL(model, dataloaders, criterion = criterion, optimizer = optimizer, epochs = int(args.epochs))
    
    validate(model, dataloaders)
    
    checkpoint = {'input_size': in_features,
              'output_size': 102,
              'epochs': args.epochs,
              'learning_rate':args.lr,
              'batch_size': 64,
              'arch': 'densenet169',
              'classifier': classifier_new,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets[0].class_to_idx
             }

    torch.save(checkpoint, 'checkpoint.pth')
    
def parse_args():
    parser = argparse.ArgumentParser(description = 'This trains a new network on a dataset of images')
    parser.add_argument('--gpu', default=False, type= bool,  help = 'Use GPU for training if available')
    parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs')
    parser.add_argument('--model', type = str, default = 'densenet169', help = 'Model architecture', choices = ['densenet169', 'vgg19'])
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate')
    parser.add_argument('--hidden_layers', type = int, default = 500, help = 'Number of hidden units')
    parser.add_argument('--data_path', action="store", type = str, default = '/home/workspace/aipnd-project/flowers', help = 'Data directory')
    
    return parser.parse_args()

def get_data(data_path):
    train_dir = data_path + '/train'
    valid_dir = data_path + '/valid'
    test_dir = data_path + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(36), 
                                          transforms.RandomResizedCrop(224), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms = [train_transforms, test_transforms]

    train_set = datasets.ImageFolder(train_dir, transform = data_transforms[0])
    valid_set = datasets.ImageFolder(valid_dir, transform = data_transforms[1])
    test_set = datasets.ImageFolder(test_dir, transform = data_transforms[1])

    image_datasets = [train_set, valid_set, test_set]

    batch_size  = 64

    trainloader = torch.utils.data.DataLoader(image_datasets[0], batch_size = batch_size, shuffle = True)
    validloader = torch.utils.data.DataLoader(image_datasets[1], batch_size = batch_size)
    testloader = torch.utils.data.DataLoader(image_datasets[2], batch_size = batch_size)

    dataloaders = [trainloader, validloader, testloader]

    return data_transforms, image_datasets,  dataloaders

def train_DL(model, dataloaders, criterion, optimizer, epochs):
    print_every = 40
    start_time = time.time()
    steps = 0
    
    running_loss = 0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for e in range(epochs):
        model.train()
        
        
        for i, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                
                validation_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    
                    for ii, (inputs, labels) in enumerate(dataloaders[1]):
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    
                        output = model.forward(inputs)
                        validation_loss += criterion(output, labels).item()
                    
                        prob = torch.exp(output)
                        equality = (labels.data == prob.max(dim = 1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                        
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders[1])),
                      "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))
                        
                running_loss = 0 
                        
                model.train()
                        
    elapsed_time = time.time() - start_time
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))
    
def validate(model, dataloaders):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    accuracy = 0

    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(dataloaders[2]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model.forward(inputs)

            prob = torch.exp(output)
            equality = (labels.data == prob.max(dim = 1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders[2])))  

main()     