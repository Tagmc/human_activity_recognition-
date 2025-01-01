import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
classes = ['handclapping','handwaving', 'Sitting', 'Standing', 'Walking', 'Walking_While_Reading_Book', 'Walking_While_Using_Phone']
num_of_timesteps = 7
num_classes = len(classes)

X, y = [], []

label = 0
for cl in classes:
    for file in os.listdir(f'/content/drive/MyDrive/dataset/{cl}'):
        print(f'Reading: /content/drive/MyDrive/dataset/{cl}/{file}')
        data = pd.read_csv(f'/content/drive/MyDrive/dataset/{cl}/{file}')
        data = data.iloc[:, 1:].values
        n_sample = len(data)
        for i in range(num_of_timesteps, n_sample):
            X.append(data[i - num_of_timesteps : i, :])
            y.append(label)
    label += 1

print("Dataset - completed")

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)


class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionRecognitionLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x = x[:, -1, :]  
        x = self.fc(x)
        return x

input_size = X.shape[2]
hidden_size = 128
model = ActionRecognitionLSTM(input_size, hidden_size, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
batch_size = 32
train_loss_history, val_loss_history = [], []
train_accuracy_history, val_accuracy_history = [], []

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss, epoch_correct = 0, 0
    
    for i in range(0, X_train.size(0), batch_size):
        optimizer.zero_grad()
        
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs, 1)
        epoch_correct += (predicted == batch_y).sum().item()
        
    epoch_loss /= X_train.size(0)
    train_loss_history.append(epoch_loss)
    train_accuracy = epoch_correct / X_train.size(0)
    train_accuracy_history.append(train_accuracy)
    
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        val_loss = criterion(outputs, y_test).item()
        val_loss_history.append(val_loss)
        
        _, predicted = torch.max(outputs, 1)
        val_accuracy = (predicted == y_test).sum().item() / X_test.size(0)
        val_accuracy_history.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Loss: {epoch_loss:.4f}, Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

torch.save(model.state_dict(), '/content/drive/MyDrive/model/model_trained_100.pth')


def visualize_loss(train_loss, val_loss, title):
    epochs = range(len(train_loss))
    plt.figure()
    plt.plot(epochs, train_loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def visualize_accuracy(train_accuracy, val_accuracy, title):
    epochs = range(len(train_accuracy))
    plt.figure()
    plt.plot(epochs, train_accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

visualize_loss(train_loss_history, val_loss_history, "Training and Validation Loss")
visualize_accuracy(train_accuracy_history, val_accuracy_history, "Training and Validation Accuracy")
