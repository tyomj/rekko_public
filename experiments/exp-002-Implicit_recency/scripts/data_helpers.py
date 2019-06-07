import pandas as pd
from datetime import datetime

def timestamp_to_dttm(ts):
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def to_timestamp(ts):
    """
    Magic numbers function to transform ts.
    """
    return (1518998400 - ((43456046.084-ts-3)/0.165))//1


def train_test_split_on_date(df: pd.DataFrame, date_split='2018-03-20'):
    df['dttm'] = df.ts.apply(to_timestamp).apply(timestamp_to_dttm)
    return df[df.dttm <= date_split], df[df.dttm > date_split]


def prepare_df_with_interactions(df: pd.DataFrame):
    df = df[['element_uid', 'user_uid']].rename({'user_uid': 'user_id', 'element_uid': 'item_id'}, axis=1)
    df = df.reset_index(drop=True)
    df['rating'] = 1
    return df


def to_universal_df_view(df: pd.DataFrame):
    return df[['user_uid', 'element_uid', 'ts', 'dttm']]


def combine_interaction_types(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    all_interaction = pd.concat([df1, df2, df3])
    all_interaction = all_interaction.sort_values(by='ts')
    all_interaction = all_interaction.reset_index(drop=True)
    return all_interaction.drop_duplicates(subset=['user_uid', 'element_uid']).reset_index(drop=True)


def recency_function(df, alpha, beta):
    df = df.rename({'user_uid': 'user_id', 'element_uid': 'item_id'}, axis=1)
    df['rating'] = df['element_launch_ts'] * alpha + df['seen_ts_since_launch'] * beta
    df = df.drop(['element_launch_ts', 'seen_ts_since_launch'], axis=1)
    df = df.reset_index(drop=True)
    return df


def csr_to_dict(csr):
    df = pd.DataFrame(list(zip(csr.nonzero()[0],
                               csr.nonzero()[1]))).rename({0: 'k', 1: 'v'}, axis=1)
    df = df.groupby('k').v.apply(list)
    d = df.to_dict()
    return d
