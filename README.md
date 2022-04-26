# DSAI_HW2
* 製造所 P96101148 巫清賢

## args
* training : training_data.csv
* testing : testing_data.csv
* output : output.csv

## 模型
* 模型 : LSTM
* 參數設定
  * kernel_initializer : glorot_uniform
  * batch_size : 30
  * Epochs : 100
* 預測值與實際值的比較
![image](https://user-images.githubusercontent.com/55480057/165135970-198e9892-d311-4e59-b226-2351c5a8c02c.png)


## 動作
* 狀態
  * BUY : 1, 買
  * NO ACTION : 0, 不動
  * SELL : -1, 賣

```python
trend = [self.get_strategy('difference', Decimal(prediction[ idx + i + 1 ]), Decimal(price)) > 0 for i in range(obs_day)]
pos_portion = int(100 * trend.count(True) / (trend.count(True) + trend.count(False)))
```
* 選擇
pos_portion分為三個層級
  * 大於70 => action : 買
  * 介於70到40 => action : 不要動作
  * 小於40 => action : 賣

## 備註
若使用requirement.txt後，依然有版本問題或其他bug，可以嘗試使用`backup/requirement.txt`，這個檔案是使用freeze自動生成的，各套件版本理論上是沒有問題。
