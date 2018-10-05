import numpy as np
import pandas as pd


def generate_dpd(df):
	df["dpd"] = df["dpd_5_cnt"]*5+df["dpd_15_cnt"]*15+df["dpd_30_cnt"]*30
	return df


def generate_dates(df):
	import itertools
	from src.data.make_dataset import date_columns
	cols = date_columns
	features = pd.DataFrame({'{}{}'.format(a, b): df[a] - df[b] for a, b in itertools.combinations(cols, 2)})
	return pd.concat([df,features], axis=1)

# Generate max of payment type
# df['Max'] = df.idxmax(axis=1)

def fix_trader_key(df):
	df["TraderKey"] = df["TraderKey"].apply(lambda x: "Trader_%s" % x)
	return df


def fix_bad_flag(df):
	df["bad_flag"] = df["bad_flag"].apply(lambda x: "Bad_%s" % x)
	return df


def fix_nans(df):
	df = df.replace(r'\s+', np.nan, regex=True)
	df = df.replace('nan', np.nan)
	return df


def fix_dpd(df):
	df["dpd_5_cnt"] = df["dpd_5_cnt"].fillna(0)
	df["dpd_15_cnt"] = df["dpd_15_cnt"].fillna(0)
	df["dpd_30_cnt"] = df["dpd_30_cnt"].fillna(0)
	return df


def fix_past_billings_cnt(df):
	df["past_billings_cnt"] = df["past_billings_cnt"].fillna(0)
	return df


def fix_scores(df):
	df["score_1"] = df["score_1"].fillna(0)
	df["score_2"] = df["score_2"].fillna(0)
	return df


def fix_dates(df, fillna=0):
	for col in df.select_dtypes(include=np.dtype('datetime64[ns]')).columns:
		df[col] = (df[col] - pd.Timestamp('2015-01-01')).dt.days
		df[col].fillna(fillna, inplace=True)
	return df

def fix_nonunique(df):
	unique_df = df.nunique().reset_index()
	unique_df.columns = ["col_name", "unique_count"]
	constant_df = unique_df[unique_df["unique_count"] == 1]
	return df.drop(columns=constant_df.col_name.tolist())