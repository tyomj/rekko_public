# The script to pickle clients raw data and create interim representation
import pandas as pd
import numpy as np

from rekko.coordinator import Coordinator
c = Coordinator()

def create_pkls():

        catalogue = c.data_raw.join('catalogue.json').load()
        catalogue = {int(k): v for k, v in catalogue.items()}
        catalogue = pd.DataFrame.from_dict(catalogue).T
        catalogue = catalogue.reset_index().rename({'index': 'element_uid'}, axis=1)

        transactions = c.data_raw.join('transactions.csv').load(dtype={
                'element_uid': np.uint16,
                'user_uid': np.uint32,
                'consumption_mode': 'category',
                'ts': np.float64,
                'watched_time': np.uint64,
                'device_type': np.uint8,
                'device_manufacturer': np.uint8
            })

        ratings = c.data_raw.join('ratings.csv').load(dtype={
                'element_uid': np.uint16,
                'user_uid': np.uint32,
                'ts': np.float64,
                'rating': np.uint8
            })

        bookmarks = c.data_raw.join('bookmarks.csv').load(dtype={
                'element_uid': np.uint16,
                'user_uid': np.uint32,
                'ts': np.float64
            })

        c.data_interim.join('catalogue.pkl').save(catalogue)
        c.data_interim.join('transactions.pkl').save(transactions)
        c.data_interim.join('ratings.pkl').save(ratings)
        c.data_interim.join('bookmarks.pkl').save(bookmarks)

if __name__ == '__main__':
    create_pkls()
