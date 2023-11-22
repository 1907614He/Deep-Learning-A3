#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tushare as ts
from sklearn.metrics import mean_squared_error
#%%
pro = ts.pro_api('e53c197d22799501016ea83f2220454f7cdd64353c7b14bbeb6c9aac')
data = pro.daily(ts_code='000001.SZ', start_date='20131111', end_date='20231111')
data.to_csv("D:/HeCloud/Master of Data Science/2023 Trimester 3/Deep Learning Fundamentals/A3/000001SZdata.csv")
#%%
# 假设 'stock_data.csv' 是你的股票数据文件
df = pd.read_csv('000001SZdata.csv')
df = df.dropna()#删除缺失值
# 这里我们只使用收盘价作为特征
data = df[['close']].values
plt.figure()
plt.plot(data,label='close')
plt.title('Time Series Plot')
plt.legend()
plt.show()

length=int(len(data)*0.85)
training_set = data[0:length]
test_set = data[length:]
# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(training_set)
test_set=scaler.transform(test_set)

# 准备数据集
def create_dataset(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60  # 使用 60 天的数据预测下一天
x, y = create_dataset(data_normalized, sequence_length)
x_train, y_train = torch.from_numpy(x).float(), torch.from_numpy(y).float()
x_, y_ = create_dataset(test_set, sequence_length)
x_test, y_test = torch.from_numpy(x_).float(), torch.from_numpy(y_).float()
#%% 构建 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x, h0.detach())
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型
model = RNN(input_size=1, hidden_size=100, num_layers=2, output_size=1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# 训练模型
num_epochs = 200
loss_count=[]
for epoch in range(num_epochs):
    outputs = model(x_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}')
        loss_count.append(loss.item())
#%%
#绘制loss图像
plt.figure()
plt.plot(loss_count,label='loss')
plt.title('Train and Validation loss')
plt.legend()
plt.show()

# 简单的预测
model.eval()
test_predict = model(x_test)
data_predict = test_predict.detach().numpy()
data_predict = scaler.inverse_transform(data_predict)  # 反归一化
y_test = scaler.inverse_transform(y_test) 
# 绘制预测结果
plt.plot(y_test, label='Actual Data')
plt.plot(data_predict, label='Predicted Data')
plt.ylabel('Price')
plt.legend()
plt.show()

##模型预测效果量化，数值越小，效果越好
#MSE 均方误差-->E[(预测值-真实值)^2]
mse=mean_squared_error(y_test,data_predict)
print('均方误差：',mse)
mse=mean_squared_error(y_test[:304],data_predict[1:])
print('均方误差：',mse)
mse=mean_squared_error(y_test[:303],data_predict[2:])
print('均方误差：',mse)
mse=mean_squared_error(y_test[:302],data_predict[3:])
print('均方误差：',mse)
