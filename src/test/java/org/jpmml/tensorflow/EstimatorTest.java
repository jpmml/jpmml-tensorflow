/*
 * Copyright (c) 2017 Villu Ruusmann
 *
 * This file is part of JPMML-TensorFlow
 *
 * JPMML-TensorFlow is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-TensorFlow is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-TensorFlow.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.tensorflow;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;

import com.google.common.base.Equivalence;
import com.google.common.base.Predicate;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ArchiveBatch;
import org.jpmml.evaluator.IntegrationTest;
import org.jpmml.evaluator.IntegrationTestBatch;
import org.tensorflow.SavedModelBundle;

abstract
public class EstimatorTest extends IntegrationTest {

	public EstimatorTest(Equivalence<Object> equivalence){
		super(equivalence);
	}

	@Override
	protected ArchiveBatch createBatch(String name, String dataset, Predicate<FieldName> predicate){
		ArchiveBatch result = new IntegrationTestBatch(name, dataset, predicate){

			@Override
			public IntegrationTest getIntegrationTest(){
				return EstimatorTest.this;
			}

			@Override
			public PMML getPMML() throws Exception {
				File savedModelDir = getSavedModelDir();

				SavedModelBundle bundle = SavedModelBundle.load(savedModelDir.getAbsolutePath(), "serve");

				try(SavedModel savedModel = new SavedModel(bundle)){
					EstimatorFactory estimatorFactory = EstimatorFactory.newInstance();

					Estimator estimator = estimatorFactory.newEstimator(savedModel);

					PMML pmml = estimator.encodePMML();

					ensureValidity(pmml);

					return pmml;
				}
			}

			private File getSavedModelDir() throws IOException, URISyntaxException {
				ClassLoader classLoader = (EstimatorTest.this.getClass()).getClassLoader();

				String protoPath = ("savedmodel/" + getName() + getDataset() + "/saved_model.pbtxt");

				URL protoResource = classLoader.getResource(protoPath);
				if(protoResource == null){
					throw new NoSuchFileException(protoPath);
				}

				File protoFile = (Paths.get(protoResource.toURI())).toFile();

				return protoFile.getParentFile();
			}
		};

		return result;
	}
}