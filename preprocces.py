import pandas as pd
import numpy as np
import haversine as hs


def preprocess(train_df, test_df, save_to_file):
    print("load dataset")
    train_df = clean_dataset(train_df)
    test_df["declared_handling_days"].fillna(train_df["declared_handling_days"].mean(), inplace=True)
    original_df, train_end_index = pd.concat([train_df, test_df]), train_df.shape[0]
    print("number of train data = " + str(train_end_index))
    feature_df = create_empty_df()
    add_binary_feature(original_df, feature_df, "b2c_c2c", "B2C")
    add_int_feature(original_df, feature_df, "seller_id")
    add_int_feature(original_df, feature_df, "long1")
    add_int_feature(original_df, feature_df, "lat1")
    add_int_feature(original_df, feature_df, "long2")
    add_int_feature(original_df, feature_df, "lat2")
    add_int_feature(original_df, feature_df, "declared_handling_days")
    add_int_feature(original_df, feature_df, "shipping_fee")
    add_int_feature(original_df, feature_df, "distance")
    add_int_feature(original_df, feature_df, "carrier_min_estimate")
    add_int_feature(original_df, feature_df, "carrier_max_estimate")
    add_average_feature(feature_df, "carrier_min_estimate", "carrier_max_estimate")
    add_int_feature(original_df, feature_df, "item_price")
    add_int_feature(original_df, feature_df, "quantity")
    add_int_feature(original_df, feature_df, "weight")
    add_categorical_feature(original_df, feature_df, "shipment_method_id")
    add_categorical_feature(original_df, feature_df, "category_id")
    add_categorical_feature(original_df, feature_df, "package_size")
    add_datetime_feature(original_df, feature_df, "acceptance_scan_timestamp")
    add_datetime_feature(original_df, feature_df, "payment_datetime")
    label = calculate_label(original_df[:train_end_index], "acceptance_scan_timestamp")
    print("preprocces finished")
    if save_to_file:
        save_df(feature_df[:train_end_index], 'final_train')
        save_df(feature_df[train_end_index:], 'final_test')
        save_df(label, 'label')
    print(list(feature_df.columns))
    return feature_df[:train_end_index], feature_df[train_end_index:], label


def save_df(df, name):
    df.to_csv("./data/" + str(name) + ".csv")


def load_dataset(train_address, test_address):
    train_df = pd.read_csv(train_address)
    train_df = clean_dataset(train_df)
    test_df = pd.read_csv(test_address)
    return pd.concat([train_df, test_df]), train_df.shape[0]


def clean_dataset(df):
    df["declared_handling_days"].fillna(df["declared_handling_days"].mean(),inplace=True)
    df = df[df["package_size"] != "NONE"]
    df = df[df["shipping_fee"] >= 0]
    df = df[df["carrier_min_estimate"] >= 0]
    df = df[df["carrier_max_estimate"] >= 0]
    df = df[df["distance"] >= 0]
    df = df[
        (pd.to_datetime(df["delivery_date"], infer_datetime_format=True) - pd.to_datetime(
            df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True)).dt.days > 0]
    df = df[
        (pd.to_datetime(df["acceptance_scan_timestamp"].str.slice(0, 10), infer_datetime_format=True) - pd.to_datetime(
            df["payment_datetime"].str.slice(0, 10), infer_datetime_format=True)).dt.days >= 0]
    df = df[
        (pd.to_datetime(df["delivery_date"].str.slice(0, 10), infer_datetime_format=True) - pd.to_datetime(
            df["payment_datetime"].str.slice(0, 10), infer_datetime_format=True)).dt.days >= 0]

    return df


def calculate_label(original_df, start_point):
    start = pd.to_datetime(original_df[start_point].str.slice(0, 10), infer_datetime_format=True)
    end = pd.to_datetime(original_df["delivery_date"], infer_datetime_format=True)
    delta = (end - start).dt.days
    print(len(delta))
    return pd.DataFrame(delta, columns=['label'])


def create_empty_df():
    return pd.DataFrame()


def add_categorical_feature(original_df, feature_df, feature_name):
    dummmies = pd.get_dummies(original_df[feature_name], prefix=feature_name)
    for col in dummmies:
        feature_df[col] = dummmies[col]
    return feature_df


def add_average_feature(feature_df, feature_name1, feature_name2):
    feature_df["carrier_average_estimate"] = feature_df[feature_name1] + feature_df[feature_name2] / 2
    return feature_df


def add_int_feature(original_df, feature_df, feature_name):
    if feature_name == "weight":
        # 2 kg --> 2.204 lbs
        feature_df[feature_name] = original_df[feature_name] * original_df["weight_units"].replace(2, 1.10231)
    else:
        feature_df[feature_name] = original_df[feature_name]

    max_value = feature_df[feature_name].max()
    min_value = feature_df[feature_name].min()
    feature_df[feature_name] = (feature_df[feature_name] - min_value) / (max_value - min_value)
    return feature_df


def add_binary_feature(original_df, feature_df, feature_name, one_value):
    feature_df[feature_name] = original_df[feature_name] == one_value


def add_datetime_feature(original_df, feature_df, feature_name):
    date_time = pd.to_datetime(original_df[feature_name].str.slice(0, 19), infer_datetime_format=True)
    # feature_df[str(feature_name) + "_day_of_week"] = date_time.dt.dayofweek
    # feature_df[str(feature_name) + "_day_of_month"] = date_time.dt.day
    # feature_df[str(feature_name) + "_month_of_year"] = date_time.dt.month
    feature_df[str(feature_name) + 'day_week_sin'] = np.sin(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_week_cos'] = np.cos(date_time.dt.dayofweek * (2. * np.pi / 7))
    feature_df[str(feature_name) + 'day_sin'] = np.sin((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'day_cos'] = np.cos((date_time.dt.day - 1) * (2. * np.pi / 31))
    feature_df[str(feature_name) + 'mnth_sin'] = np.sin((date_time.dt.month - 1) * (2. * np.pi / 12))
    feature_df[str(feature_name) + 'mnth_cos'] = np.cos((date_time.dt.month - 1) * (2. * np.pi / 12))
    return None


def add_zipcode(data_name):
    df = pd.read_csv("./data/" + data_name + ".tsv", sep="\t")
    df = df.iloc[:1000000]
    df['distance'] = ""
    df['long1'] = ""
    df['long2'] = ""
    df['lat1'] = ""
    df['lat2'] = ""

    for index, row in df.iterrows():
        if index % 100000 == 0:
            print("procces " + str(index) + " data sucsessfully")

        if str.isdigit(str(row['buyer_zip'])[:5]):
            buyer_zip_coor = z.get(int(row['buyer_zip'][:5]))
        else:
            buyer_zip_coor = None

        if str.isdigit(str(row['item_zip'])[:5]):
            item_zip_coor = z.get(int(row['item_zip'][:5]))
        else:
            item_zip_coor = None

        if (buyer_zip_coor is not None) and (item_zip_coor is not None):
            distance = round(hs.haversine(buyer_zip_coor, item_zip_coor))
            long1 = buyer_zip_coor[0]
            lat1 = buyer_zip_coor[1]
            long2 = item_zip_coor[0]
            lat2 = item_zip_coor[1]

        else:
            distance = -100
        df.at[index, 'distance'] = distance
        df.at[index, 'long1'] = long1
        df.at[index, 'lat1'] = lat1
        df.at[index, 'long2'] = long2
        df.at[index, 'lat2'] = lat2
    df.at[index, 'distance'] = distance
    df.at[index, 'long1'] = long1
    df.at[index, 'lat1'] = lat1
    df.at[index, 'long2'] = long2
    df.at[index, 'lat2'] = lat2

    return df


zip_df = pd.read_csv("./data/zipcode.txt", sep='\t', header=None)
lat_long = tuple(zip(zip_df[9].tolist(), zip_df[10].tolist()))
zip_code = zip_df[1].tolist()
z = {zip_code[i]: lat_long[i] for i in range(len(zip_df.index))}

train_df = add_zipcode("train")
test_df = add_zipcode("quiz")
# print(np.unique(train_df["declared_handling_days"]))

X, x_quiz, y = preprocess(train_df, test_df, True)
print(np.unique(x_quiz["declared_handling_days"]))

