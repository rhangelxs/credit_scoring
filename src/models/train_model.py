import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

class CatBoostCustomModel:
	def __init__(self, model, model_params={}):
		self.result = {}
		try:
			self.model = model
			self.model.set_params(**model_params)
		except:
			raise

	def fit(self, X, y, cat_features=[], fit_params={}, n_folds=None, shuffle=False):
		if not len(cat_features):
			cat_features = np.where(X.dtypes == np.object)[0]
		if not n_folds:
			df_pool = Pool(X, label=y, cat_features=cat_features)
			model_fit = self.model.fit(df_pool, **fit_params)
		else:
			kf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=shuffle)
			model_fit = self.model
			for train_index, valid_index in tqdm(kf.split(X, y)):
				print(len(train_index), len(valid_index))
				df_pool = Pool(X.iloc[train_index], label=y.iloc[train_index], cat_features=cat_features)
				x_val, y_val = X.iloc[valid_index], y.iloc[valid_index]

				model_fit.fit(df_pool, eval_set=(x_val, y_val)
				              , **fit_params)
		self.result = {
			"X": X,
			"y": y,
			"cat_features": cat_features,
			"model": self.model,
			"fit_params": fit_params,
			"model_fit": model_fit
		}
		return model_fit

	def get_features_importance(self, sorted=False):
		feature_score = pd.DataFrame(
			list(zip(self.result.get("X").dtypes.index, self.result.get("model_fit").get_feature_importance())),
			columns=['Feature', 'Score'])
		if sorted:
			feature_score.sort_values(by='Score', ascending=False, inplace=True, na_position='last')
		return feature_score

	def plot_features_importance(self, sorted=True):
		feature_score = self.get_features_importance(sorted=sorted)
		ax = feature_score.plot('Feature', 'Score', kind='bar')
		ax.set_title("Catboost Feature Importance Ranking", fontsize=14)
		ax.set_xlabel('')

		rects = ax.patches

		# get feature score as labels round to 2 decimal
		labels = feature_score['Score'].round(2)

		for rect, label in zip(rects, labels):
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width() / 2, height + 0.35, label, ha='center', va='bottom')

		plt.show()

	def get_score(self):
		return self.result["model_fit"].score(self.result["X"], self.result["y"])

	def get_crosstab(self):
		crosstab = pd.DataFrame()
		crosstab['GroundTruth'] = self.result["y"]
		crosstab['Predict'] = self.result["model_fit"].predict(self.result["X"])
		return pd.crosstab(crosstab['GroundTruth'], crosstab['Predict'], margins=True)
