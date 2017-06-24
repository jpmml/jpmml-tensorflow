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

import com.google.common.primitives.Floats;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CMatrixUtil;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.framework.NodeDef;

abstract
public class LinearEstimator extends Estimator {

	public LinearEstimator(SavedModel savedModel, String head){
		super(savedModel, head);
	}

	public RegressionModel encodeRegressionModel(TensorFlowEncoder encoder){
		SavedModel savedModel = getSavedModel();

		NodeDef biasAdd = savedModel.getOnlyInput(getHead(), "BiasAdd");

		int count;

		{
			Operation operation = savedModel.getOperation(biasAdd.getName());

			Output output = operation.output(0);

			long[] shape = ShapeUtil.toArray(output.shape());
			if((shape.length != 2) || (shape[0] != -1)){
				throw new IllegalArgumentException();
			}

			count = (int)shape[1];
		}

		List<Equation> equations = new ArrayList<>();

		for(int i = 0; i < count; i++){
			Equation equation = new Equation();

			equations.add(equation);
		}

		NodeDef addN = savedModel.getOnlyInput(biasAdd.getInput(0), "AddN");

		List<String> inputNames = addN.getInputList();
		for(String inputName : inputNames){
			NodeDef term = savedModel.getOnlyInput(inputName, "MatMul", "Select");

			// "real_valued_column"
			if(("MatMul").equals(term.getOp())){
				NodeDef placeholder = savedModel.getNodeDef(term.getInput(0));
				NodeDef multiplier = savedModel.getOnlyInput(term.getInput(1), "VariableV2");

				Feature feature = encoder.createContinuousFeature(savedModel, placeholder);

				try(Tensor tensor = savedModel.run(multiplier.getName())){
					float[] values = TensorUtil.toFloatArray(tensor);

					for(int i = 0; i < count; i++){
						Equation equation = equations.get(i);

						equation.addTerm(feature, floatToDouble(values[i]));
					}
				}
			} else

			// "sparse_column_with_keys"
			if(("Select").equals(term.getOp())){
				NodeDef placeholder = savedModel.getOnlyInput(term.getInput(0), "Placeholder");
				NodeDef findTable = savedModel.getOnlyInput(term.getInput(1), "LookupTableFind");
				NodeDef multiplier = savedModel.getOnlyInput(term.getInput(2), "VariableV2");

				Map<?, ?> table = savedModel.getTable(findTable.getInput(0));

				List<String> categories = (List)new ArrayList<>(table.keySet());

				List<? extends Feature> features = encoder.createBinaryFeatures(savedModel, placeholder, categories);

				float[] values;

				try(Tensor tensor = savedModel.run(multiplier.getName())){
					values = TensorUtil.toFloatArray(tensor);
				}

				for(int i = 0; i < equations.size(); i++){
					Equation equation = equations.get(i);

					List<Float> categoryValues = CMatrixUtil.getColumn(Floats.asList(values), features.size(), equations.size(), i);

					for(int j = 0; j < features.size(); j++){
						Feature feature = features.get(j);

						int index = ValueUtil.asInt((Number)table.get(categories.get(j)));

						equation.addTerm(feature, floatToDouble(categoryValues.get(index)));
					}
				}
			} else

			{
				throw new IllegalArgumentException(term.getName());
			}
		}

		NodeDef bias = savedModel.getOnlyInput(biasAdd.getInput(1), "VariableV2");

		try(Tensor tensor = savedModel.run(bias.getName())){
			float[] values = TensorUtil.toFloatArray(tensor);

			for(int i = 0; i < count; i++){
				Equation equation = equations.get(i);

				equation.setIntercept(floatToDouble(values[i]));
			}
		}

		RegressionModel regressionModel = new RegressionModel();

		for(Equation equation : equations){
			RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(equation.getFeatures(), equation.getCoefficients(), equation.getIntercept());

			regressionModel.addRegressionTables(regressionTable);
		}

		return regressionModel;
	}

	static
	private class Equation {

		private List<Feature> features = new ArrayList<>();

		private List<Double> coefficients = new ArrayList<>();

		private Double intercept = null;


		private Equation(){
		}

		public void addTerm(Feature feature, Double coefficient){
			this.features.add(feature);
			this.coefficients.add(coefficient);
		}

		public List<Feature> getFeatures(){
			return this.features;
		}

		public List<Double> getCoefficients(){
			return this.coefficients;
		}

		public Double getIntercept(){
			return this.intercept;
		}

		public void setIntercept(Double intercept){
			this.intercept = intercept;
		}
	}
}