from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import csv
import numpy as np
from tensorflow.keras.losses import MSE, MAE
from sklearn.model_selection import train_test_split


def calculate_delivery_date(quiz_label):
    quiz_data = pd.read_csv("./data/quiz.tsv", sep="\t")
    quiz_data = pd.to_datetime(quiz_data["acceptance_scan_timestamp"].str.slice(0, 10))
    out_file = open('./data/final_predict_RF.csv', 'w+', newline='')
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for index, value in quiz_data.items():
        tsv_writer.writerow([str(15000001 + index), str(value + pd.Timedelta(days=quiz_label[index]))[:10]])
        if index % 100000 == 0:
            print(index)
    out_file.flush()
    out_file.close()


def load_dataset(train_address, test_address, label_address):
    X = pd.read_csv("./data/" + train_address)
    x_quiz = pd.read_csv("./data/" + test_address)
    y = pd.read_csv("./data/" + label_address)['label']
    return X, x_quiz, y


X, x_quiz, y = load_dataset("final_train.csv", "final_test.csv", "label.csv")
print("load dataset")
X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.001)
##### Training Phase ####
print("train random forest model begin")
model = RandomForestRegressor(verbose=2,n_estimators=500,min_samples_split=8,max_features=6)
model.fit(train_X, train_y)
print(model.score(test_X,test_y))

pred_test = model.predict(test_X)
pred_train = model.predict(train_X)

test_mse = MSE(test_y, pred_test)
test_mae = MAE(test_y, pred_test) / 2
print("TEST MSE : % f" % (test_mse))
print("TEST MAE : % f" % (test_mae))
train_mse = MSE(train_y, pred_train)
train_mae = MAE(train_y, pred_train) / 2
print("TRAIN MSE : % f" % (train_mse))
print("TRAIN MAE : % f" % (train_mae))

result_df = pd.DataFrame(model.predict(x_quiz))
result_df.to_csv("./data/quiz_result_RF.csv", header=None)
# result_df = pd.read_csv("./data/quiz_result_RF.csv", header=None)
calculate_delivery_date(result_df[1].values.round())
