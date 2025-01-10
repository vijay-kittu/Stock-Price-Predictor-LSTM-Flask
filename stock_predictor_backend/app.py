from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load and preprocess the stock data (replace with your actual file path)
def load_data(filepath, ticker):
    data = pd.read_csv(filepath)
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data['Date'] = pd.to_datetime(ticker_data['Date'])
    ticker_data.sort_values('Date', inplace=True)

    scaler = MinMaxScaler()
    ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(
        ticker_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    )
    if ticker_data.empty:
        return jsonify({'error': 'Ticker not found in dataset'}), 404

    return ticker_data, scaler

# Create dataset for LSTM model
class StockDataset(Dataset):
    def __init__(self, data, target_col, window_size=60):
        self.data = data
        self.target_col = target_col
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.window_size, :].values
        y = self.data.iloc[idx + self.window_size][self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define the LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

# Predict future stock price using the trained model
def predict(model, data, window_size, device):
    model.eval()
    with torch.no_grad():
        last_sequence = data.iloc[-window_size:, :].values
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(last_sequence).item()
    return prediction

@app.route('/predict', methods=['POST'])
def predict_stock_price():
    ticker = request.json.get('ticker')
    future_days = request.json.get('future_days', 1)

    # Load model (replace with your trained model path)
    model = StockLSTM(input_size=5).to('cpu')
    model.load_state_dict(torch.load('stock_model.pth', map_location='cpu'))  # Load your trained model

    # Load the data
    filepath = 'C:/users/vijay/Downloads/kaggle_projects/stock_market_data_tickers/stock_predictor_backend/stock_data.csv' 

    data, scaler = load_data(filepath, ticker)

    # Make the prediction
    window_size = 60
    future_price = predict(model, data[['Open', 'High', 'Low', 'Close', 'Volume']], window_size, 'cpu')
    future_price = scaler.inverse_transform([[0, 0, 0, future_price, 0]])[0][3]

    return jsonify({'predicted_price': future_price})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
