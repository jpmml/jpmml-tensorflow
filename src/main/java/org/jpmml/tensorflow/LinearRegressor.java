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
import java.util.List;
import java.util.Map;

import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Label;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.tensorflow.Tensor;
import org.tensorflow.framework.NodeDef;

public class LinearRegressor extends Estimator {

	public LinearRegressor(SavedModel savedModel){
		this(savedModel, LinearRegressor.HEAD);
	}

	public LinearRegressor(SavedModel savedModel, String head){
		super(savedModel, head);
	}

	@Override
	public RegressionModel encodeModel(TensorFlowEncoder encoder){
		SavedModel savedModel = getSavedModel();

		NodeDef biasAdd = savedModel.getOnlyInput(getHead(), "BiasAdd");

		Label label;

		{
			DataField dataField = encoder.createDataField(FieldName.create("_target"), OpType.CONTINUOUS, DataType.FLOAT);

			label = new ContinuousLabel(dataField);
		}

		List<Feature> features = new ArrayList<>();

		List<Double> coefficients = new ArrayList<>();

		NodeDef addN = savedModel.getOnlyInput(biasAdd.getInput(0), "AddN");

		List<String> inputNames = addN.getInputList();
		for(String inputName : inputNames){
			NodeDef term = savedModel.getOnlyInput(inputName, "MatMul", "Select");

			if(("MatMul").equals(term.getOp())){
				NodeDef multiplicand = savedModel.getNodeDef(term.getInput(0));
				NodeDef multiplier = savedModel.getOnlyInput(term.getInput(1), "VariableV2");

				Feature feature = encodeContinuousFeature(multiplicand, encoder);

				features.add(feature);

				try(Tensor tensor = savedModel.run(multiplier.getName())){
					float value = TensorUtil.toFloatScalar(tensor);

					coefficients.add(floatToDouble(value));
				}
			} else

			if(("Select").equals(term.getOp())){
				NodeDef multiplicand = savedModel.getOnlyInput(term.getInput(0), "Placeholder");
				NodeDef findTable = savedModel.getOnlyInput(term.getInput(1), "LookupTableFind");
				NodeDef multiplier = savedModel.getOnlyInput(term.getInput(2), "VariableV2");

				Map<?, ?> table = savedModel.getTable(findTable.getInput(0));

				List<String> categories = new ArrayList(table.keySet());

				Feature feature = encodeCategoricalFeature(multiplicand, categories, encoder);

				float[] values;

				try(Tensor tensor = savedModel.run(multiplier.getName())){
					values = TensorUtil.toFloatArray(tensor);
				}

				DataField dataField = (DataField)encoder.getField(feature.getName());

				for(String category : categories){
					BinaryFeature binaryFeature = new BinaryFeature(encoder, dataField, category);

					features.add(binaryFeature);

					int index = ValueUtil.asInt((Number)table.get(category));

					coefficients.add(floatToDouble(values[index]));
				}
			} else

			{
				throw new IllegalArgumentException();
			}
		}

		Double intercept;

		{
			NodeDef bias = savedModel.getOnlyInput(biasAdd.getInput(1), "VariableV2");

			try(Tensor tensor = savedModel.run(bias.getName())){
				float value = TensorUtil.toFloatScalar(tensor);

				intercept = floatToDouble(value);
			}
		}

		Schema schema = new Schema(label, features);

		RegressionModel regressionModel = new RegressionModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(schema), null)
			.addRegressionTables(RegressionModelUtil.createRegressionTable(schema.getFeatures(), intercept, coefficients));

		return regressionModel;
	}

	static
	private double floatToDouble(float value){
		return Double.parseDouble(Float.toString(value));
	}

	public static final String HEAD = "linear/regression_head/predictions/scores";
}