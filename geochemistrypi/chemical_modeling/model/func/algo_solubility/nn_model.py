import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNet(nn.Module):
    def __init__(self, inp_shape, activation=None):
        super(NeuralNet, self).__init__()
        self.activation = self.get_activation(activation)
        self.fc1 = nn.Linear(inp_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 8)
        self.fc_out = nn.Linear(8, 1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc_out(x)
        return x

    @staticmethod
    def get_activation(activation):
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())


class NN:
    def __init__(self, x_train, x_test, y_train, y_test, df_pred, inp_shape, activation=None, lr=None):
        self.df_pred = torch.tensor(df_pred, dtype=torch.float32)
        self.x_train, self.x_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
        self.y_train, self.y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
        self.inp_shape = inp_shape
        self.activation = activation
        self.lr = lr

    def parameterize_model(self):
        model_nn = NeuralNet(self.inp_shape, self.activation)
        return model_nn

    def optimize_model(self):
        model_nn = self.parameterize_model()
        optimizer = optim.Adam(model_nn.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        return model_nn, optimizer, criterion

    def fit_nn(self, model, optimizer, criterion, epochs=100, batch_size=5, evaluation=False):
        dataset = TensorDataset(self.x_train, self.y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(x_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        if evaluation:
            model.eval()
            with torch.no_grad():
                test_pred = model(self.x_test).squeeze()
                nn_loss = criterion(test_pred, self.y_test).item()
                rmse = torch.sqrt(torch.mean((test_pred - self.y_test) ** 2)).item()
            return nn_loss, rmse

        return model

    def evaluate_model(self, evaluation=False):
        model, optimizer, criterion = self.optimize_model()
        model = self.fit_nn(model, optimizer, criterion, evaluation=evaluation)
        with torch.no_grad():
            nn_pred = model(self.x_test).numpy()
            nn_pred_df = model(self.df_pred).numpy()
        return nn_pred, nn_pred_df

    def plot_results(self, epochs=100):
        _, y_pred = self.evaluate_model(epochs=epochs)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(11, 7))
        plt.scatter(self.y_test.numpy(), y_pred)
        plt.xlabel("True C test values")
        plt.ylabel("Predictions on the test set")
        plt.title("NN model results")
        plt.grid(False)
        plt.show()


class NNP(NN):
    def __init__(self, x_train, x_test, y_train, y_test, df_pred, inp_shape, activation=None, lr=None):
        super().__init__(x_train, x_test, y_train, y_test, df_pred, inp_shape, activation, lr)

    def predict_nn(self, df_test):
        model, _, _ = self.optimize_model()
        model = self.fit_nn(model, None, None, evaluation=False)
        df_test = torch.tensor(df_test, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(df_test).numpy()
        return y_pred
