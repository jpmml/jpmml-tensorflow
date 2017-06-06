JPMML-TensorFlow
================

Java library and command-line application for converting [TensorFlow](http://tensorflow.org) models to PMML.

# Features #

* Supported Estimator types:
  * [`learn.DNNClassifier`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
  * [`learn.DNNRegressor`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNRegressor)
  * [`learn.LinearClassifier`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearClassifier)
  * [`learn.LinearRegressor`](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearRegressor)
* Supported Feature column types:
  * [`layers.one_hot_column`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/one_hot_column)
  * [`layers.real_valued_column`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/real_valued_column)
  * [`layers.sparse_column_with_keys`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_keys)
* Production quality:
  * Complete test coverage.
  * Fully compliant with the [JPMML-Evaluator](https://github.com/jpmml/jpmml-evaluator) library.

# Prerequisites #

### The TensorFlow side of operations

* Protocol Buffers 3.2.0 or newer
* TensorFlow 1.1.0 or newer

### The Java side of operations

* Java 1.8 or newer

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/); use the `protoc.exe` system property to specify the location of the Protocol Buffers compiler:
```
mvn -Dprotoc.exe=/usr/local/bin/protoc clean install
```

The build produces an executable uber-JAR file `target/converter-executable-1.0-SNAPSHOT.jar`.

# Usage #

A typical workflow can be summarized as follows:

1. Use TensorFlow to train an estimator.
2. Export the estimator in `SavedModel` data format to a directory in a local filesystem.
3. Use the JPMML-TensorFlow command-line converter application to turn the SavedModel directory to a PMML file.

### The TensorFlow side of operations

Please see the test script file [main.py](https://github.com/jpmml/jpmml-tensorflow/blob/master/src/test/resources/main.py) for sample workflows.

### The Java side of operations

Converting the estimator SavedModel directory `estimator/` to a PMML file `estimator.pmml`:
```
java -jar target/converter-executable-1.0-SNAPSHOT.jar --tf-savedmodel-input estimator/ --pmml-output estimator.pmml
```

Getting help:
```
java -jar target/converter-executable-1.0-SNAPSHOT.jar --help
```

# License #

JPMML-TensorFlow is licensed under the [GNU Affero General Public License (AGPL) version 3.0](http://www.gnu.org/licenses/agpl-3.0.html). Other licenses are available on request.

# Additional information #

Please contact [info@openscoring.io](mailto:info@openscoring.io)
