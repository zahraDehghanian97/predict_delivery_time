# import logging
# import xgboost
# import xgboost as xgb
# from sklearn.ensemble import GradientBoostingRegressor
import logging
import pandas as pd
import csv
import numpy as np
from tensorflow.keras.losses import MSE, MAE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def calculate_delivery_date():
    quiz_label = pd.read_csv("./output/quiz_result.csv", header=None).round()[0].to_list()
    quiz_data = pd.read_csv("./data/quiz.tsv", sep="\t")
    quiz_data = pd.to_datetime(quiz_data["acceptance_scan_timestamp"].str.slice(0, 10))
    out_file = open('./output/output.tsv', 'w+', newline='')
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for index, value in quiz_data.items():
        tsv_writer.writerow([str(15000001 + index), str(value + pd.Timedelta(days=quiz_label[index]))[:10]])
        if index % 100000 == 0:
            print(index)
    out_file.flush()
    out_file.close()

def preprocess(train_address, test_address, save_to_file):
    logging.info("1")
    original_df, train_end_index = load_dataset(train_address, test_address)
    logging.info("2")
    logging.info("train index: " + str(train_end_index))
    feature_df = create_empty_df()
    logging.info("3")
    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    logging.info("4")
    add_int_feature(original_df, feature_df, "long1")
    add_int_feature(original_df, feature_df, "lat1")
    add_int_feature(original_df, feature_df, "long2")
    add_int_feature(original_df, feature_df, "lat2")
    add_int_feature(original_df, feature_df, "declared_handling_days")
    logging.info("5")
    add_int_feature(original_df, feature_df, "shipping_fee")
    logging.info("5.5")
    add_int_feature(original_df, feature_df, "distance")
    logging.info("6")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    logging.info("7")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    logging.info("8")
    add_int_feature(original_df, feature_df, "item_price")
    logging.info("9")
    add_int_feature(original_df, feature_df, "quantity")
    logging.info("10")
    add_int_feature(original_df, feature_df, "weight")
    logging.info("11")
    add_categorical_feature(original_df, feature_df, "shipment_method_id")
    logging.info("12")
    add_categorical_feature(original_df, feature_df, "category_id")
    logging.info("13")
    add_categorical_feature(original_df, feature_df, "package_size")
    logging.info("14")
    add_datetime_feature(original_df, feature_df, "acceptance_scan_timestamp")
    add_datetime_feature(original_df, feature_df, "payment_datetime")
    logging.info("15")
    label = calculate_label(original_df[:train_end_index], "acceptance_scan_timestamp")
    logging.info("15")
    if save_to_file:
        save_df(feature_df, 'x')
    logging.info("16")
    if save_to_file:
        save_df(label, 'y')
    logging.info(list(feature_df.columns))
    return feature_df[:train_end_index], feature_df[train_end_index:], label


def save_df(df, name):
    df.to_csv(str(name) + ".csv")


def load_dataset(train_address, test_address):
    train_df = pd.read_pickle(train_address)
    #train_df = train_df[:1000]
    train_df = clean_dataset(train_df)
    test_df = pd.read_pickle(test_address)#, sep="\t")
    return pd.concat([train_df, test_df]), train_df.shape[0]


def clean_dataset(df):
    logging.info("29")
    df["declared_handling_days"].fillna(df["declared_handling_days"].mean())
    logging.info("30")
    df = df[df["shipping_fee"] >= 0]
    logging.info("31")
    df = df[df["carrier_min_estimate"] >= 0]
    logging.info("32")
    df = df[df["carrier_max_estimate"] >= 0]
    df = df[df["distance"] >= 0]
    df = df[
        (pd.to_datetime(df["delivery_date"], infer_datetime_format=True) - pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True)).dt.days > 0]
    logging.info("34")
    df = df[
        (pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True) - pd.to_datetime(
            df["payment_datetime"].str.slice(0, 10), infer_datetime_format=True)).dt.days >= 0]

    return df


def calculate_label(original_df, start_point):
    start = pd.to_datetime(original_df[start_point].str.slice(0, 10), infer_datetime_format=True)
    logging.info("100")
    end = pd.to_datetime(original_df["delivery_date"], infer_datetime_format=True)
    logging.info("101")
    delta = (end - start).dt.days
    return delta


def create_empty_df():
    return pd.DataFrame()


def add_time_feature(original_df, feature_df, feature_name):
    return feature_df


def add_categorical_feature(original_df, feature_df, feature_name):
    dummmies = pd.get_dummies(original_df[feature_name], prefix=feature_name)
    for col in dummmies:
        feature_df[col] = dummmies[col]

    return feature_df


def add_int_feature(original_df, feature_df, feature_name):
    feature_df[feature_name] = original_df[feature_name]

    if feature_name == "weight":
        # 2 kg --> 2.204 lbs
        feature_df[feature_name] = original_df[feature_name] * original_df["weight_units"].replace(2, 2.20462)

    return feature_df


def add_binary_feature(original_df, feature_df, feature_name, one_value):
    feature_df[feature_name] = original_df[feature_name] == one_value


def add_datetime_feature(original_df, feature_df, feature_name):
    logging.info("30")
    date_time = pd.to_datetime(original_df[feature_name].str.slice(0, 19), infer_datetime_format=True)
    logging.info("31")
    feature_df[str(feature_name) + "_day_of_week"] = date_time.dt.dayofweek
    logging.info("33")
    feature_df[str(feature_name) + "_day_of_month"] = date_time.dt.day
    logging.info("34")
    feature_df[str(feature_name) + "_month_of_year"] = date_time.dt.month

    return None


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

X, x_quiz, y = preprocess("./data/train_w_zip.pkl", "./data/quiz_w_zip.pkl", False)
X = np.asarray(X).astype('float32')
x_quiz = np.asarray(x_quiz).astype('float32')

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                    test_size=0.001)
##### Training Phase ####
model = XGBRegressor(n_estimators=1000, max_depth=8, verbosity=2, tree_method='gpu_hist')

model.fit(train_X, train_y, eval_set=[(test_X, test_y)],
        eval_metric='mae',
        verbose=True)
pred = model.predict(test_X)
pred_test = model.predict(train_X)
test_mse = MSE(test_y, pred)
test_mae = MAE(test_y, pred)/2
print("TEST MSE : % f" %(test_mse))
print("TEST MAE : % f" %(test_mae))

train_mse = MSE(train_y, pred_test)
train_mae = MAE(train_y, pred_test)/2
print("TRAIN MSE : % f" %(train_mse))
print("TRAIN MAE : % f" %(train_mae))
np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")



# normalizer = preprocessing.Normalization(axis=-1)
# normalizer.adapt(np.array(X))
# model = build_and_compile_model(normalizer)
# history = train_model(model, X, y)
# logging.info("start saving result for quiz set")
# np.savetxt("./output/quiz_result.csv", model.predict(x_quiz), delimiter=",")
#########################


calculate_delivery_date()
logging.info("finished")

# dtrain = xgb.DMatrix(X, label=y)
# dtest = xgb.DMatrix(x_quiz)
# param = {}
# param['tree_method'] = 'gpu_hist'
# param['verbosity'] = 2
# bst = xgb.train(param, dtrain, verbose_eval=True, num_boost_round=100)
# np.savetxt("./data/quiz_result.csv", bst.predict(dtest), delimiter=",")