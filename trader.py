# system
import os

# basic
import numpy as np
import pandas as pd

# visual
import matplotlib.pyplot as plt

# sklearn evaluate
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# calculate
from decimal import Decimal

class Predict():
    def __init__(self, train_filename, test_filename, output_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.output_filename = output_filename

        self.train_data = None
        self.test_data = None
        
        self.scaler = MinMaxScaler(feature_range=(0, 1)) #正規化，值介於0-1

        self.stock = 0

    # 載入訓練資料跟測試資料，並新增其title
    def loadData(self):
        train_data = pd.read_csv(self.train_filename, sep = ',', names=["Open","High","Low","Close"])
        test_data = pd.read_csv(self.test_filename, sep = ',', names=["Open","High","Low","Close"])
        return train_data,test_data

    def preprocessData(self):
        #擷取Open欄位的data
        train_open = self.train_data.iloc[:, 0:1].values

        #正規化
        train_open_scaled = self.scaler.fit_transform(train_open)
        # print(train_open_scaled)

        # Feature selection
        xtrain = []
        ytrain = []
        for i in range(1, len(train_open_scaled)):
            xtrain.append(train_open_scaled[i - 1 : i, 0])
            ytrain.append(train_open_scaled[i, 0])

        xtrain, ytrain = np.array(xtrain), np.array(ytrain)
        # print(xtrain)
        # print(ytrain)
        xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1], 1))
        # print(xtrain)
        return xtrain, ytrain

    # build LSTM model
    def buildModel(self, input_size):
        model = Sequential()
        model.add(LSTM(units=900, return_sequences = True, kernel_initializer = 'glorot_uniform', input_shape  =  input_size))
        model.add(Dropout(0.3))
        model.add(LSTM(units = 900, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model

    def predictModel(self):
        # testing data ready
        test_open = self.test_data.iloc[:, 0:1].values #taking  open price
        total = pd.concat([self.train_data['Open'], self.test_data['Open']], axis=0)
        locus = len(total) - len(self.test_data) - 1
        test_input = self.scaler.transform(total[locus:].values.reshape(-1,1))
        xtest = np.array([test_input[i - 1 : i, 0] for i in range(1, 21)]) #creating input for lstm prediction
        xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
        
        # predicting
        predicted_value = self.scaler.inverse_transform(self.model.predict(xtest))
        
        # evaluate
        print(mean_squared_error(test_open, predicted_value, squared = False))

        # 以下程式碼取消註解可以取得test and predict比較圖
        # self.figure_output(test_data = test_open, predict_data = predicted_value)
        
        return [value[0] for value in predicted_value.tolist()]

    # test and predict figure
    def figure_output(self, test_data, predict_data):
        plt.figure(figsize=(10, 5))
        plt.plot(test_data,'red', label = 'Test Prices')
        plt.plot(predict_data, 'blue', label = 'Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        plt.title('Test vs Predicted Prices')
        plt.legend(loc = 'best', fontsize = 20)
        plt.show()

    def get_status(self, status):
        return {
            'BUY': 1,
            'NO ACTION': 0,
            'SELL': -1
        }[status]
        
    def making_action(self, msg):
        return {
            'BUY': 'NO ACTION' if self.stock == 1 else 'BUY',
            'NO ACTION': 'NO ACTION',
            'SELL': 'NO ACTION' if self.stock == -1 else 'SELL',
        }[msg]
        
    def get_strategy(self, strategy, x_1, x_2):
        return {
            'portion': (x_2 - x_1) / (x_2 + x_1),
            'difference': (x_2 - x_1) / (x_2 + x_1),
        }[strategy]


    def main(self):
        # training data ready
        self.train_data, self.test_data = self.loadData()
        xtrain, ytrain = self.preprocessData()
        
        # model ready
        self.model = self.buildModel(input_size = (xtrain.shape[1], xtrain.shape[2]))
        
        # model training
        self.model.fit(xtrain, ytrain, batch_size=30, epochs=100)

        # prediction
        prediction = self.predictModel()
        
        # decision making 
        prediction_num = len(prediction)
        stock_num = 0
        opt_text = ''
        obs_day = 3
        for idx, price in enumerate(prediction):
            action = 0
            if idx + 1 <= (prediction_num - obs_day):
                trend = [self.get_strategy('difference', Decimal(prediction[ idx + i + 1 ]), Decimal(price)) > 0 for i in range(obs_day)]
                pos_portion = int(100 * trend.count(True) / (trend.count(True) + trend.count(False)))
                if pos_portion > 70:
                    action = self.get_status(self.making_action('BUY'))
                elif 70 >= pos_portion > 40:
                    action = self.get_status(self.making_action('NO ACTION'))
                else:
                    action = self.get_status(self.making_action('SELL'))
            else:
                if stock_num != 0:
                    action = self.get_status(self.making_action('SELL' if stock_num == 1 else 'BUY'))
                else:
                    action = self.get_status(self.making_action('NO ACTION'))
            self.stock += action
            if idx + 1 != prediction_num:
                opt_text += f'{action}\n'
            
        with open(self.output_filename, 'w') as f:
            f.writelines(opt_text)
            f.close()


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')

    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')

    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    predict = Predict(train_filename=args.training, test_filename=args.testing, output_filename=args.output)
    predict.main()
