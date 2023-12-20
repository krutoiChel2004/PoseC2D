import torch
import torch.nn as nn
import albumentations as A

from tqdm import tqdm

from model import PoseC2D
from data import YogaPoses

list_path = ['data\\YogaPoses\\Downdog',
             'data\\YogaPoses\\Goddess',
             'data\\YogaPoses\\Plank',
             'data\\YogaPoses\\Tree',
             'data\\YogaPoses\\Warrior2']

transform_train = A.Compose(
    [
        A.Resize(640, 640)
    ]
)

def get_data(image_dir, batch_size=16, transform=None, shuffle=True, pin_memory=False):
    data = YogaPoses(image_dir, transform=transform)
    train_size = int(0.8 * data.__len__())
    test_size = data.__len__() - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_batch,test_batch

data_train,data_test = get_data(list_path, batch_size=16, transform=transform_train)


device = 'cuda'
model = PoseC2D().to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def accuracy(pred, label):
    _, pred = torch.max(pred, dim=1)
    pred = pred.reshape(-1, 1)
    label = label.reshape(-1, 1)
    answer = pred.cpu().numpy() == label.cpu().numpy()
    return answer.mean()

epochs = 1

torch.cuda.empty_cache()

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(data_train)):
        X, y = sample['img'], sample['class']
        # print(len(X))
        X = X.to(device)
        y = y.to(device)
        
        label = y
        pred = model(X)
        
        loss = loss_fn(pred, label)
        # print(pred)
        # print(torch.max(pred,dim=1))
        # print(label)
        loss.backward()
        loss_item = loss.item()
        loss_val += loss_item
        
        optimizer.step()
        optimizer.zero_grad()
        acc_current = accuracy(pred, label)
        acc_val += acc_current 
        pbar.set_postfix(loss=f'{loss_item:.5f}', accuracy=f'{acc_current:.3f}')
        
    print(f'epoch: {epoch + 1}')    
    print(f'loss: {loss_val/len(data_train)}')
    print(f'acc: {acc_val/len(data_train)}')

torch.save(model.state_dict(), 'weights\\PoseC2D\\poseC2D_weightsV1.pth')