import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(Autoencoder, self).__init__()
        # Encoder: Maps input to embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)  # Embedding layer
        )
        # Decoder: Maps embedding back to original input size
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

def train_autoencoder(model, data, epochs=100, batch_size=32, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, reconstruction = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")