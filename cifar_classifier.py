from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
import torchvision
from torchvision import datasets, models, transforms
from torch.backends import cudnn
import matplotlib.pyplot as plt
import time
import os
from random import random
import copy
from  utils import imsave
from logger import Logger
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For fast training.
cudnn.benchmark = True



def get_loader(data_dir, eval_type='gan_train', mode = 'train'):

    # Data augmentation and normalization for training
    # Just normalization for validation
    
    if eval_type == 'gan_train':
    
        
        data_transforms = {
            'train': transforms.Compose([       
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            
            'infer': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
        
        
    if eval_type == 'gan_test':
    
        data_transforms = {
            'train': transforms.Compose([       
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            
            'infer': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
                
                
    if mode == 'train':
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                     shuffle=True, num_workers=128)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

    else : 
    
        test_image_datasets = datasets.ImageFolder( data_dir,data_transforms['infer'])
        print (test_image_datasets.classes)
        dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = 32, shuffle= False, num_workers = 32)
        dataset_sizes = len(test_image_datasets)
        class_names = test_image_datasets.classes
    
    return dataloaders, class_names, dataset_sizes
        
        
    
    
    
    

def visualize_save_image(data_dir):

    ######### visualize the images in grid ##########
    
    dataloaders, class_names, _ = get_loader(data_dir = data_dir)
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imsave(out,'img.png')
    
    
#### function to plot confusion matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #transpose the matrix to make x-axis True Class and Y-axis Predicted Class
    cm= np.transpose(cm)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           #here we are not printing the title
           #title=title,
           xlabel='True label',
           ylabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig,ax



def model():
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features    
    model_ft.fc = nn.Linear(num_ftrs, 100)
    
    return model_ft

def train_model(output_dir, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    logger = Logger(os.path.join(output_dir, 'log_dir'))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            
                    
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                
            if phase =='train':
                tag='train'
            if phase == 'val':  
                tag='val' 
                
            logger.scalar_summary(tag, epoch_loss, epoch)
            
            
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # save the best model
                PATH = os.path.join(output_dir,'cifar_classifier.pth')
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc
                        }, PATH)
            
                  

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
def evaluate_classification_err (model, checkpoint_path, dataloaders, dataset_sizes, criterion):
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    label_list=[]
    prediction_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            #print (i,labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            label_list.extend(labels.tolist())
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            prediction_list.extend(preds.tolist())
            
            
            loss = criterion(outputs, labels)
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        avg_loss = running_loss / dataset_sizes
        avg_acc = running_corrects.double() / dataset_sizes

            
        return avg_loss, avg_acc.item(), label_list, prediction_list


    
def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
    
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
def train(data_dir,output_dir, eval_type):
       
    
    model_ft = model()
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    dataloaders, class_names, dataset_sizes = get_loader(data_dir, eval_type, 'train')
    train_model(output_dir,model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
   
   
def cls_err(data_dir, output_dir,eval_type):
    model_ft= model()
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    dataloaders, class_names, dataset_sizes = get_loader(data_dir=data_dir, eval_type= eval_type, mode='inference')
    checkpoint = os.path.join(output_dir,'cifar_classifier.pth')
    err, acc, label_list, prediction_list = evaluate_classification_err(model_ft, checkpoint, dataloaders, dataset_sizes, criterion)
    print (err, acc)
    print (classification_report(label_list, prediction_list, target_names=class_names))
    fig,ax= plot_confusion_matrix(label_list,prediction_list, class_names,title='Confusion Matrix')
    plt.show()
    filename = 'confusion.png'
    fig.savefig(filename)


parser = argparse.ArgumentParser()


#required arguments
parser.add_argument('--output_dir', type=str, default = './outputs/')
parser.add_argument('--data_dir', type=str, default='/volume3/AAM-GAN/stargan_rafd/train_results')
parser.add_argument('--mode', type=str, default = 'test', choices=['train','test'])
parser.add_argument('--eval_type', type=str, default= 'gan_test', choices=['gan_train','gan_test'])
config = parser.parse_args()
    
#visualize the data
#visualize_save_image(data_dir = '/volume3/AAM-GAN/detected_faces' )

if config.mode == 'train':
    # train the model                   
    train(config.data_dir, config.output_dir, config.eval_type)
    
if config.mode =='test':
    #find the classification err and accuracy
    cls_err(config.data_dir, config.output_dir, config.eval_type)

