import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pickle
import os 


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR100(root='./data_org', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=1)

testset = torchvision.datasets.CIFAR100(root='./data_org', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=1)
                                         
image_save_dir = './data'

    
# create class directories if doesnot exist
org_cls = [str(x) for x in range (100)]
for cls in org_cls:
    if not os.path.exists(os.path.join(image_save_dir,'train', cls)):
       os.makedirs(os.path.join(image_save_dir, 'train', cls))
       
    if not os.path.exists(os.path.join(image_save_dir,'val', cls)):
       os.makedirs(os.path.join(image_save_dir, 'val', cls))
                                         
for i, (x_real, c_org) in enumerate(trainloader):
                   
    
    # Prepare input images and target domain labels.
    image_name = str(i)+'.png'    
    
    
    #save org image to input dir of results_dir
    img = x_real.cpu()
    save_image(img,os.path.join(image_save_dir,'train', str(c_org.item()),image_name) , nrow=1, padding=0)
    print ('saving image.....', image_name)
    
for i, (x_real, c_org) in enumerate(testloader):
                   

    
    # Prepare input images and target domain labels.
    image_name = str(i)+'.png'      
    
    #save org image to input dir of results_dir
    
    img = x_real.cpu()
    save_image(img,os.path.join(image_save_dir,'val', str(c_org.item()),image_name) , nrow=1, padding=0)
    print ('saving image.....', image_name)


