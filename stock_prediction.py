import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow import keras

# Load dataset
df = pd.read_csv('GE.csv')

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Select columns for training (excluding Date)
cols = list(df)[1:6]
df_for_training = df[cols].astype(float)

# Normalize the dataset
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

# Prepare training data
n_future = 1  # Number of days we want to predict into the future
n_past = 14   # Number of past days used for training

trainX, trainY = [], []

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, :])  
    trainY.append(df_for_training_scaled[i + n_future - 1, 0])  

trainX, trainY = np.array(trainX), np.array(trainY)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer

# Compile model
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

# Train the model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Plot loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

# Forecasting future values
n_future = 25  
forecast_period_dates = pd.date_range(df['Date'].iloc[-1], periods=n_future, freq='D')

forecast = model.predict(trainX[-n_future:])
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]

# Convert timestamp to dates
df_forecast = pd.DataFrame({'Date': np.array(forecast_period_dates), 'Open': y_pred_future})

# Prepare original dataset for plotting
original = df[['Date', 'Open']].copy()  # Use .copy() to avoid SettingWithCopyWarning
original = original[original['Date'] >= '2021-03-28']

# Plot results
sns.lineplot(x=original['Date'], y=original['Open'], label="Actual")
sns.lineplot(x=df_forecast['Date'], y=df_forecast['Open'], label="Forecasted")
plt.xticks(rotation=45)
plt.show()
