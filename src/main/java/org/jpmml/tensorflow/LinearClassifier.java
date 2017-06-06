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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;

public class LinearClassifier extends LinearEstimator {

	public LinearClassifier(SavedModel savedModel, String head){
		super(savedModel, head);
	}

	@Override
	public RegressionModel encodeModel(TensorFlowEncoder encoder){
		DataField dataField = encoder.createDataField(FieldName.create("_target"), OpType.CATEGORICAL, DataType.INTEGER);

		RegressionModel regressionModel = encodeRegressionModel(encoder);

		List<RegressionTable> regressionTables = regressionModel.getRegressionTables();

		List<String> categories;

		if(regressionTables.size() == 1){
			categories = Arrays.asList("0", "1");

			RegressionTable passiveRegressionTable = new RegressionTable(0)
				.setTargetCategory(categories.get(0));

			regressionModel.addRegressionTables(passiveRegressionTable);

			RegressionTable activeRegressionTable = regressionTables.get(0)
				.setTargetCategory(categories.get(1));
		} else

		if(regressionTables.size() > 2){
			categories = new ArrayList<>();

			for(int i = 0; i < regressionTables.size(); i++){
				RegressionTable regressionTable = regressionTables.get(i);
				String category = String.valueOf(i);

				regressionTable.setTargetCategory(category);

				categories.add(category);
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		dataField = encoder.toCategorical(dataField.getName(), categories);

		CategoricalLabel categoricalLabel = new CategoricalLabel(dataField);

		regressionModel
			.setMiningFunction(MiningFunction.CLASSIFICATION)
			.setNormalizationMethod(RegressionModel.NormalizationMethod.SOFTMAX)
			.setMiningSchema(ModelUtil.createMiningSchema(categoricalLabel))
			.setOutput(ModelUtil.createProbabilityOutput(categoricalLabel));

		return regressionModel;
	}

	public static final String BINARY_LOGISTIC_HEAD = "linear/binary_logistic_head/predictions/probabilities";
	public static final String MULTI_CLASS_HEAD = "linear/multi_class_head/predictions/probabilities";
}