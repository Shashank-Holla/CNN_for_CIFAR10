import torch
import torchvision
import torchvision.transforms as transforms

def train(net, device, trainloader, optimizer, criterion, epoch):
    net.train()  
    running_loss = 0.0
    epoch_train_loss = 0.0
    epoch_train_accuracy = 0
    correct = 0
    processed = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs. Data is not on cuda. Move it. Number of images, labels per iteration = batch size in transform
        inputs, labels = data
              
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad() 

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #Loss calculation is on cuda
        loss.backward()
        optimizer.step()

        #Accuracy calculation
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        processed += len(inputs)
        

        # print statistics
        running_loss += loss.item()
        epoch_train_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # print("Number of correct",correct,"\nTotal:",processed)    
    epoch_train_accuracy=(100*correct/processed)
    # print("Accuracy=",epoch_train_accuracy)
    epoch_train_loss /= i
    # print("Total loss for epoch: ",i, "is", epoch_train_loss)

    # print('Finished Training')
    
    return epoch_train_accuracy, epoch_train_loss