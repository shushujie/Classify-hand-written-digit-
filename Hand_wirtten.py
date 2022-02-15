#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8
#Name: Shu Huang UCLA ID: 604743235
#PartI: Logistic Regression
#Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, 
  shuffle=False,sampler=SubsetRandomSampler(subset_indices.view(-1)))


# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 10

#useful Hyper Parameters
input_dim = 28 * 28
num_classes = 1
learningRate = 0.01
Momentum = 0.9


#Create model to implement logistic regression as a derived subclass of torch.nn.Module, where the forward pass is also given.
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__() # inJava3:super().__init__()
        # Instanciate nn.Module class and assign Linear as a member
        self.linear = nn.Linear(input_dim, num_classes)
        #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # write the sequence of layers and processes
        # x -> Linear -> output
        out = self.linear(x)       
        return out
#Initialize a model
model = LogisticRegression(input_dim, num_classes)

#Create Loss function
class LR_loss(nn.modules.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        batch_size = 64 #outputs.size[0]
        loss = torch.sum(torch.log( 1 + torch.exp(-labels * outputs.t()))
                        ) / batch_size #loss function
        return loss
#set up the loss function
criterion = LR_loss()

#Setting the optimizer using Stochastic gradient descent algorithm
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate,  momentum = Momentum)


# Training the Model
print('Logistic Regression:')
#for epoch in range(num_epochs):
for epoch in range(num_epochs):
    total_loss = 0
    loss_sum = 0
    for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            labels = Variable(2*(labels.float()-0.5))
            #labels = labels.type(torch.LongTensor)
            
            #do forward pass and backward and optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ## Print your results every epoch
           
            loss_sum = loss_sum + loss.item()
            
            if (i+1) % len(train_loader) == 0:
               
                average_loss = loss_sum / (i + 1)
                #print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'% (epoch + 1, len(train_loader), i + 1, num_epochs, loss.item()) )   
                print('Epoch: [% d/% d], Batch ID: [% d/% d], Averaged Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_loader), average_loss) #the num_epochs should be len(train_data) // batch_size, if do this generally,
                     )   
            

 
# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.view(-1, 28*28)    
    ## Put your prediction code here, currently use a random prediction
    testOutputs = torch.sigmoid(model(images))
    #print(testOutputs.shape)
    prediction = testOutputs.data >= 0.5    
    #print(prediction.shape)
    correct += (prediction.view(-1).long() == labels).sum()#prediction.view(-1).long()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


 
#Create model to implement SVM as a derived subclass of torch.nn.Module, where the forward pass is also given.
class SVMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVMModel, self).__init__() # inJava3:super().__init__()
        # Instanciate nn.Module class and assign Linear as a member
        self.linear = nn.Linear(input_dim, num_classes)
        #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # write the sequence of layers and processes
        # x -> Linear -> output
        out = self.linear(x)       
        return out
#Initialize a model
model = SVMModel(input_dim, num_classes)

#Create Loss function
class SVM_loss(nn.modules.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, labels):
        batch_size = 64
        loss = torch.sum(torch.clamp( 1 - labels * outputs.t() , min = 0)
                        ) / batch_size #loss function
        return loss
#set up the loss function
criterion = SVM_loss()

#Setting the optimizer using Stochastic gradient descent algorithm
optimizer = torch.optim.SGD(model.parameters(), lr = learningRate,  momentum = Momentum)

 
print('SVM:')
# Training the Model
#for epoch in range(num_epochs):
for epoch in range(num_epochs):
    total_loss = 0
    loss_sum = 0
    for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28*28)
            labels = Variable(2*(labels.float()-0.5))
            #labels = labels.type(torch.LongTensor)
            
            #do forward pass and backward and optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ## Print your results every epoch
           
            loss_sum = loss_sum + loss.item()
            
            if (i+1) % len(train_loader) == 0:
               
                average_loss = loss_sum / (i + 1)
                #print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'% (epoch + 1, len(train_loader), i + 1, num_epochs, loss.item()) )   
                print('Epoch: [% d/% d], Batch ID: [% d/% d], Averaged Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_loader), average_loss) #the num_epochs should be len(train_data) // batch_size, if do this generally,
                     )   
            




# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.view(-1, 28*28)    
    ## Put your prediction code here, currently use a random prediction
    testOutputs = torch.sigmoid(model(images))
    #print(testOutputs.shape)
    prediction = testOutputs.data >= 0.5    
    #print(prediction.shape)
    correct += (prediction.view(-1).long() == labels).sum()#prediction.view(-1).long()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


        




