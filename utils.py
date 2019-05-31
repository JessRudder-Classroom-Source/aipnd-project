import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image

def create_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
    
    print("The loaders have been created...")
    return trainloader, testloader, validloader, train_data

# Set up the initial values for our model parameters
def set_up_model_params(architecture, learning_rate, hidden_units, device):
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    arch_methods = {
        'densenet': models.densenet121,
        'resnet': models.resnet18,
        'alexnet': models.alexnet,
        'vgg': models.vgg16
    }

    arch_inputs = {
        'densenet': 1024,
        'resnet': 512,
        'alexnet': 9216,
        'vgg': 25088
    }
    
    model = arch_methods[architecture](pretrained=True)
    input_size = arch_inputs[architecture]

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                     nn.ReLU(),
                                     nn.Linear(hidden_units, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    model.to(device)
    
    print("Your model structure has been instantiated...")
    return model, device, criterion, optimizer

# This is where the magic happens, call this and your model gets trained
def train_the_model(model, trainloader, validloader, criterion, optimizer, device, epochs):
    print("The training of your model will now commence!")
    epochs = epochs
    train_losses, valid_losses = [], []

    with active_session():
        for e in range(epochs):
            running_loss = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            else:
                valid_loss = 0
                accuracy = 0

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    model.eval()
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model(inputs)
                        valid_loss += criterion(output, labels)

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                model.train()

                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

# Save the checkpoint so we don't have to retrain the model every time
def save_model(model, optimizer, save_dir, architecture, train_data):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    arch_inputs = {
        'densenet': 1024,
        'resnet': 512,
        'alexnet': 9216,
        'vgg': 25088
    }

    checkpoint = {'architecture': architecture,
                  'input_size': arch_inputs[architecture],
                  'output_size': 102,
                  'features': model.features,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'idx_to_class': model.class_to_idx
                 }

    torch.save(checkpoint, save_dir + 'checkpoint.pth')
    print("The model has been saved.")

# Validate the accuracy of your trained model using the previously unused test data
def validate_model(model, testloader, device):
    model.to(device)
    # Using Test set to validate the accuracy of the model   
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        model.eval()

        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: %d %%' % (100 * correct_predictions / total_predictions))

# Load a saved checkpoint
def load_checkpoint(filepath, architecture, hidden_units):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_info = torch.load(filepath)
    arch_map = {
        'densenet': models.densenet121,
        'resnet': models.resnet18,
        'alexnet': models.alexnet,
        'vgg': models.vgg16
    }
    arch_inputs = {
        'densenet': 1024,
        'resnet': 512,
        'alexnet': 9216,
        'vgg': 25088
    }
    
    model = arch_map[architecture](pretrained=True)
    input_size = arch_inputs[architecture]
    default_classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )
    classifier = model_info.get('classifier', default_classifier)

    model.classifier = classifier
    model.class_to_idx = model_info.get('idx_to_class')
    model.load_state_dict(model_info['state_dict'])
    model.to(device)
    
    return model

# Process image to prepare it to be classified
def process_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image)

    preprocess = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Your image is being preprocessed...")

    return preprocess(image)

def predict(image_path, model, topk, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to("cpu")
    preprocessed_image = process_image(image_path)
    
    preprocessed_image.unsqueeze_(0)
    preprocessed_image.requires_grad_(False)
    preprocessed_image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(preprocessed_image)
        results = torch.exp(output).topk(topk)
    
    probs = results[0][0]
    classes = results[1][0]
    
    return probs, classes