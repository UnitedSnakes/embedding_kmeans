import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from autoencoder import Autoencoder


def load_embeddings(fold_num, input_folder="results/bert_embeddings"):
    input_path = f"{input_folder}/bert_embeddings_fold{fold_num}.csv"
    df = pd.read_csv(input_path)
    
    df['raw_embeddings'] = df['raw_embeddings'].apply(eval)
    
    embeddings = torch.tensor(df['raw_embeddings'].tolist(), dtype=torch.float32)
    return df, embeddings


def process_autoencoder(
    fold_num,
    input_folder="results/bert_embeddings",
    output_folder="results/bert_embeddings_reduced",
    input_dim=768,  # BERT-base-cased's output dimension
    hidden_dim=256,
    bottleneck_dim=64,  # target dimension for reduced embeddings
    epochs=50,
    learning_rate=0.001,
):
    df, embeddings = load_embeddings(fold_num, input_folder)
    
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        encoded, decoded = model(embeddings)
        loss = loss_fn(decoded, embeddings)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        reduced_embeddings, _ = model(embeddings)
    
    df['reduced_embeddings'] = reduced_embeddings.tolist()
    
    df.drop(columns=["raw_embeddings"], inplace=True)
    
    output_path = f"{output_folder}/bert_embeddings_reduced_fold{fold_num}.csv"
    
    df.to_csv(output_path, index=False)

    print(f"Processed fold {fold_num} and saved reduced embeddings to {output_path}")


if __name__ == "__main__":
    for i in range(1, Config.num_folds + 1):
        process_autoencoder(i)
