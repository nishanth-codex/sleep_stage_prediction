
import torch.nn as nn
import torch
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, d_in,d_h,d_out):
        super(SimpleModel, self).__init__()
        self.hidden1 = nn.Linear(d_in, d_h[0])  # Hidden layer: 2x3000 inputs -> 128 neurons

        self.hidden_layers = []
        for i in range(len(d_h)-1):
            self.hidden_layers.append(nn.Linear(d_h[i],d_h[i+1]))

        self.output = nn.Linear(d_h[-1], d_out)   # Output layer: 128 neurons -> 6 output
        self.relu = nn.ReLU()           # Activation functions
        
    def forward(self, x):
        h = self.relu(self.hidden(x))
        for hidden in self.hidden_layers:
            h = hidden(h)
        h_ = self.relu(h)
        y = self.output(h)
        return y

    def train_model(self, X_train, y_train , n_epochs = 300):
        self.train()
        criterion = nn.CrossEntropyLoss() # Cross Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=0.0005)
        for epoch in range(n_epochs):
            predictions = self.forward(X_train)
            loss = criterion(predictions,y_train)

            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients (backpropagation)
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        return self 

    def test_model(self, X_test, y_test):
        self.eval()    
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            y_pred = self(X_test)
            loss = criterion(y_pred,y_test)

            predicted_classes = torch.argmax(y_pred, dim=1)
            accuracy = (predicted_classes == y_test).float().mean()

            print(f"Test Loss: {loss.item():.4f}")
            print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

