import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import logging
import os

class ForecastModel:
    def __init__(self, data_path, n_input=5, n_features=1, datapoints=12):
        self.data_path = data_path
        self.n_input = n_input
        self.n_features = n_features
        self.datapoints = datapoints
        self.scaler = MinMaxScaler()
        self.category_forecasts = {}

        # Configure logging
        log_folder = 'log'
        os.makedirs(log_folder, exist_ok=True)
        log_filename = os.path.join(log_folder, f'forecast_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        self.logger.info('Loading data from %s', self.data_path)
        df1 = pd.read_csv(self.data_path, parse_dates=['Timestamp'])
        return df1

    def prepare_data(self, df, category):
        self.logger.info('Preparing data for category %s', category)
        df_category = df[df['Category'] == category]
        df_category = df_category.drop(columns=['Category', 'Forecast'])
        df_category.columns = ['Date', 'Production']
        df_category['Date'] = pd.to_datetime(df_category['Date'])
        df_category.set_index('Date', inplace=True)
        df_category.dropna(inplace=True)
        return df_category

    def scale_data(self, train, test):
        self.logger.info('Scaling data')
        self.scaler.fit(train)
        scaled_train = self.scaler.transform(train)
        scaled_test = self.scaler.transform(test)
        return scaled_train, scaled_test

    def create_generators(self, scaled_train):
        self.logger.info('Creating data generators')
        generator = TimeseriesGenerator(scaled_train, scaled_train, length=self.n_input, batch_size=1)
        train_size = int(len(scaled_train) * 0.8)
        train_generator = TimeseriesGenerator(scaled_train[:train_size], scaled_train[:train_size], length=self.n_input, batch_size=1)
        valid_generator = TimeseriesGenerator(scaled_train[train_size:], scaled_train[train_size:], length=self.n_input, batch_size=1)
        return generator, train_generator, valid_generator

    def build_model(self):
        self.logger.info('Building LSTM model')
        model = Sequential()
        model.add(Input(shape=(self.n_input, self.n_features)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, model, train_generator, valid_generator):
        self.logger.info('Training model')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, restore_best_weights=True)
        model.fit(train_generator, epochs=50, validation_data=valid_generator, verbose=1)#, callbacks=[early_stopping])

    def make_predictions(self, model, scaled_train, scaled_test, test):
        self.logger.info('Making predictions')
        test_predictions = []
        first_eval_batch = scaled_train[-self.n_input:]
        current_batch = first_eval_batch.reshape((1, self.n_input, self.n_features))

        for i in range(len(test)):
            current_pred = model.predict(current_batch, verbose=0)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        true_predictions = self.scaler.inverse_transform(test_predictions)
        predicted_df = pd.DataFrame(data=true_predictions, index=test.index, columns=['Predictions'])

        future_predictions = []
        current_batch = scaled_test[-self.n_input:].reshape((1, self.n_input, self.n_features))

        for i in range(12):
            current_pred = model.predict(current_batch, verbose=0)[0]
            future_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        future_predictions = self.scaler.inverse_transform(future_predictions)
        future_dates = pd.date_range(start=test.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
        future_predicted_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Future Predictions'])

        return predicted_df, future_predicted_df

    def plot_results(self, category, train, test, predicted_df, future_predicted_df):
        self.logger.info('Plotting results for category %s', category)
        plt.figure(figsize=(14, 8))
        plt.plot(train.index, train['Production'], label='Training Data')
        plt.plot(test.index, test['Production'], label='Test Data')
        plt.plot(predicted_df.index, predicted_df['Predictions'], label='Predictions')
        plt.plot(future_predicted_df.index, future_predicted_df['Future Predictions'], label='Future Predictions')
        plt.title(f'Category {category}')
        plt.legend()
        plt.savefig(f'forecast_{category}.png')

    def forecast(self):
        self.logger.info('Starting forecast process')
        df1 = self.load_data()
        categories = df1['Category'].unique()

        for category in categories:
            df_category = self.prepare_data(df1, category)
            train = df_category.iloc[:-self.datapoints]
            test = df_category.iloc[-self.datapoints:]
            scaled_train, scaled_test = self.scale_data(train, test)
            generator, train_generator, valid_generator = self.create_generators(scaled_train)
            model = self.build_model()
            self.train_model(model, train_generator, valid_generator)
            predicted_df, future_predicted_df = self.make_predictions(model, scaled_train, scaled_test, test)

            self.category_forecasts[category] = {
                'train': train,
                'test': test,
                'predicted_df': predicted_df,
                'future_predicted_df': future_predicted_df
            }

            self.plot_results(category, train, test, predicted_df, future_predicted_df)

if __name__ == "__main__":
    forecast_model = ForecastModel(data_path='input.csv')
    forecast_model.forecast()