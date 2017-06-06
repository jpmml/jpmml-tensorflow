from pandas import DataFrame
from tensorflow.contrib.learn import DNNClassifier, DNNRegressor, LinearClassifier, LinearRegressor
from tensorflow.contrib.layers import one_hot_column, real_valued_column, sparse_column_with_keys
from tensorflow.contrib.layers.python.layers.feature_column import _OneHotColumn, _RealValuedColumn, _SparseColumnKeys
from tensorflow.contrib.learn.python.learn.utils.input_fn_utils import InputFnOps

import numpy
import pandas
import os
import shutil
import tempfile
import tensorflow as tf

def load_csv(name):
	return pandas.read_csv("csv/" + name)

def store_csv(df, name):
	df.to_csv("csv/" + name, index = False)

def store_savedmodel(estimator, serving_input_fn, name):
	savemodel_dir = estimator.export_savedmodel(tempfile.mkdtemp(), serving_input_fn = serving_input_fn, as_text = True)
	savemodel_dir = savemodel_dir.decode("UTF-8")

	if(os.path.isdir("savedmodel/" + name)):
		shutil.rmtree("savedmodel/" + name)
	shutil.move(savemodel_dir, "savedmodel/" + name)

def _dnn_feature_columns(feature_columns):
	return list(map(lambda x: one_hot_column(x) if isinstance(x, _SparseColumnKeys) else x, feature_columns))

def _input_fn(df, cont_feature_columns, cat_feature_columns, label_column):
	cont_features = {column : tf.constant(df[column].values, dtype = tf.float64, shape = [df[column].size, 1]) for column in cont_feature_columns}
	cat_features = {column : tf.constant(df[column].values, dtype = tf.string, shape = [df[column].size, 1]) for column in cat_feature_columns}
	features = dict(list(cont_features.items()) + list(cat_features.items()))
	label = tf.constant(df[label_column].values, shape = [df[label_column].size, 1])
	return features, label

def _serving_input_fn(cont_feature_columns, cat_feature_columns):
	cont_feature_placeholders = {column : tf.placeholder(dtype = tf.float64, shape = [None, 1], name = column) for column in cont_feature_columns}
	cat_feature_placeholders = {column : tf.placeholder(dtype = tf.string, shape = [None, 1], name = column) for column in cat_feature_columns}
	feature_placeholders = dict(list(cont_feature_placeholders.items()) + list(cat_feature_placeholders.items()))
	features = {column : tensor for column, tensor in feature_placeholders.items()}
	label = None
	return InputFnOps(features, label, feature_placeholders)

#
# Binary classification
#

audit_df = load_csv("Audit.csv")
audit_df["Adjusted"] = audit_df["Adjusted"].astype(int)

audit_cont_columns = ["Age", "Income", "Deductions", "Hours"]
audit_cat_columns = ["Employment", "Education", "Marital", "Occupation", "Gender"]

audit_feature_columns = [real_valued_column(column, dtype = tf.float64) for column in audit_cont_columns] + [sparse_column_with_keys(column, dtype = tf.string, keys = sorted(audit_df[column].unique())) for column in audit_cat_columns]

def audit_input_fn():
	return _input_fn(audit_df, audit_cont_columns, audit_cat_columns, "Adjusted")

def audit_serving_input_fn():
	return _serving_input_fn(audit_cont_columns, audit_cat_columns)

def build_audit(classifier, name, with_proba = True):
	classifier.fit(input_fn = audit_input_fn, steps = 2000)

	adjusted = DataFrame(classifier.predict(input_fn = audit_input_fn, as_iterable = False), columns = ["_target"])
	if(with_proba):
		adjusted_proba = DataFrame(classifier.predict_proba(input_fn = audit_input_fn, as_iterable = False), columns = ["probability(0)", "probability(1)"])
		adjusted = pandas.concat((adjusted, adjusted_proba), axis = 1)
	store_csv(adjusted, name + ".csv")

	store_savedmodel(classifier, audit_serving_input_fn, name)

build_audit(DNNClassifier(hidden_units = [71, 11], feature_columns = _dnn_feature_columns(audit_feature_columns)), "DNNClassificationAudit")
build_audit(LinearClassifier(feature_columns = audit_feature_columns), "LinearClassificationAudit")

#
# Multi-class classification
#

iris_df = load_csv("Iris.csv")
iris_df["Species"] = iris_df["Species"].replace("setosa", "0").replace("versicolor", "1").replace("virginica", "2").astype(int)

iris_cont_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]

iris_feature_columns = [real_valued_column(column, dtype = tf.float64) for column in iris_cont_columns]

def iris_input_fn():
	return _input_fn(iris_df, iris_cont_columns, [], "Species")

def iris_serving_input_fn():
	return _serving_input_fn(iris_cont_columns, [])

def build_iris(classifier, name, with_proba = True):
	classifier.fit(input_fn = iris_input_fn, steps = 500)

	species = DataFrame(classifier.predict(input_fn = iris_input_fn, as_iterable = False), columns = ["_target"])
	if(with_proba):
		species_proba = DataFrame(classifier.predict_proba(input_fn = iris_input_fn, as_iterable = False), columns = ["probability(0)", "probability(1)", "probability(2)"])
		species = pandas.concat((species, species_proba), axis = 1)
	store_csv(species, name + ".csv")

	store_savedmodel(classifier, iris_serving_input_fn, name)

build_iris(DNNClassifier(hidden_units = [11], feature_columns = _dnn_feature_columns(iris_feature_columns), n_classes = 3), "DNNClassificationIris")
build_iris(LinearClassifier(feature_columns = iris_feature_columns, n_classes = 3), "LinearClassificationIris")

#
# Regression
#

auto_df = load_csv("Auto.csv")
auto_df["origin"] = auto_df["origin"].astype(str)

auto_cont_columns = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
auto_cat_columns = ["origin"]

auto_feature_columns = [real_valued_column(column, dtype = tf.float64) for column in auto_cont_columns] + [sparse_column_with_keys(column, dtype = tf.string, keys = sorted(auto_df[column].unique())) for column in auto_cat_columns]

def auto_input_fn():
	return _input_fn(auto_df, auto_cont_columns, auto_cat_columns, "mpg")

def auto_serving_input_fn():
	return _serving_input_fn(auto_cont_columns, auto_cat_columns)

def build_auto(regressor, name):
	regressor.fit(input_fn = auto_input_fn, steps = 2000)

	mpg = DataFrame(regressor.predict(input_fn = auto_input_fn, as_iterable = False), columns = ["_target"])
	store_csv(mpg, name + ".csv")

	store_savedmodel(regressor, auto_serving_input_fn, name)

build_auto(DNNRegressor(hidden_units = [7, 5, 3], feature_columns = _dnn_feature_columns(auto_feature_columns)), "DNNRegressionAuto")
build_auto(LinearRegressor(feature_columns = auto_feature_columns), "LinearRegressionAuto")
