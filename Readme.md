# LSTM-Based Stock Price Prediction

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical stock market data. The model is trained using past stock prices and aims to forecast future values.

## Features
- Loads historical stock data from a CSV file.
- Preprocesses and normalizes the dataset.
- Uses an LSTM neural network to learn patterns and predict future prices.
- Visualizes actual vs. predicted stock prices.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas keras tensorflow scikit-learn matplotlib seaborn
```

## Dataset
The dataset should be in CSV format and contain at least the following columns:
- `Date`: The date of the stock data.
- `Open`: The opening price.
- `High`: The highest price of the day.
- `Low`: The lowest price of the day.
- `Close`: The closing price.
- `Volume`: The trading volume.

## How to Use
1. **Prepare the Dataset**: Ensure your CSV file (e.g., `GE.csv`) is in the working directory.
2. **Run the Script**: Execute the Python script to train the LSTM model.
3. **View Predictions**: The script will generate a plot comparing actual vs. predicted stock prices.

Run the script using:
```bash
python stock_prediction.py
```

## Model Architecture
- LSTM layer with 64 units (relu activation, return sequences enabled)
- LSTM layer with 32 units (relu activation, return sequences disabled)
- Dropout layer (20% dropout rate)
- Dense output layer

## Results
The model is trained for 10 epochs with a batch size of 16. A visualization of actual vs. predicted stock prices is generated using Matplotlib and Seaborn.

## Example Output
After training, a plot is generated showing actual stock prices and predicted future prices:
![Example Plot](example_plot.png)

## Future Improvements
- Fine-tuning hyperparameters to improve accuracy.
- Adding additional stock indicators as input features.
- Using a more complex architecture with attention mechanisms.

## License
This project is open-source and available under the MIT License.

