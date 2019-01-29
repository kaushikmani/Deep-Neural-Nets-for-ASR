import torch
import torch.optim as optim
import Dataset
import CNN
import RNN
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class Train():
    def __init__(self):
            #CNN parameters
            feature_dim = 246
            label_dim = 64
            num_filters = 128
            kernel_size = 5
            drop_rate = 0.5

            #RNN parameters
            # feature_dim = 246
            # hidden_size = 128
            # num_layers = 1
            # dropout_rate = 0.5
            # label_dim = 64
            # biDirectional = True
            min_loss = float("inf")
            self.isBest = False

            train_dataset = Dataset.Dataset('train_factory.mat')
            dataset_size = len(train_dataset)
            validation_split = 0.2
            random_seed = 42
            shuffle_dataset = True
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
            self.validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=valid_sampler)


            test_dataset = Dataset.Dataset('test_factory.mat')
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
            self.total_step = len(self.train_loader)
            self.model = CNN.CNNModel(feature_dim=feature_dim,label_dim = label_dim, num_filters = num_filters, kernel_size=kernel_size, drop_rate=drop_rate)
            #model = RNN.RNNModel(feature_dim=feature_dim, hidden_size=128, num_layers, drop_rate, label_dim, biDirectional=True)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.num_epochs = 50
            for epoch in range(self.num_epochs):
                self.train_dataset(epoch)
                val_loss = self.val_dataset()

                self.isBest = val_loss < min_loss
                min_loss = min(val_loss, min_loss)

            if self.isBest:
                torch.save(self.model, "model_CNN.pt")

            # Test the model
            if self.isBest:
                self.model = torch.load("model_CNN.pt")
            self.model.eval()
            loss = 0
            with torch.no_grad():
                for i, (test_feature, test_label) in enumerate(self.test_loader):
                    outputs = self.model(test_feature)
                    loss = loss + self.model.loss(outputs, test_label)
            print("Average Testing Loss:", loss.item()/len(self.test_loader))

    def train_dataset(self, epoch):
        for i, (feature, label) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    #Forward pass
                    outputs = self.model(feature)
                    loss = self.model.loss(outputs, label)

                    #Backward pass
                    loss.backward()
                    self.optimizer.step()

                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, self.num_epochs, i+1, self.total_step, loss.item()))

    def val_dataset(self):
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for i, (feature, label) in enumerate(self.validation_loader):
                output = self.model(feature)
                loss = loss + self.model.loss(output, label)
        val_loss = loss.item()/len(self.validation_loader)
        return val_loss


def main():
    train = Train()


if __name__ == '__main__':
    main()
