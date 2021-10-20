import logging

import haversine as hs
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
zip_df = pd.read_csv("./data/zipcode.txt", sep='\t', header=None)
lat_long = tuple(zip(zip_df[9].tolist(), zip_df[10].tolist()))
zip_code = zip_df[1].tolist()
z = {zip_code[i]: lat_long[i] for i in range(len(zip_df.index))}

data_name = "quiz"
df = pd.read_csv("./data/" + data_name + ".tsv", sep="\t")

df['distance'] = ""
df['long1'] = ""
df['long2'] = ""
df['lat1'] = ""
df['lat2'] = ""

counter = 0
logging.info("start")
for index, row in df.iterrows():
    if index % 10000 == 0:
        logging.info(index)

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
        #print(str(counter), "error for ", str(row['buyer_zip'])[:5], str(row['item_zip'])[:5])
        #counter += 1

    df.at[index, 'distance'] = distance

    df.at[index, 'long1'] = long1
    df.at[index, 'lat1'] = lat1

    df.at[index, 'long2'] = long2
    df.at[index, 'lat2'] = lat2

df.to_pickle("./data/" + data_name + "_w_zip.pkl")
df.to_csv("./data/" + data_name + "_w_zip.tsv")