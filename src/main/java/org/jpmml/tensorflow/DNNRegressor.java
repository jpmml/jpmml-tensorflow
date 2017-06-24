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

import java.util.List;

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.ContinuousLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;

public class DNNRegressor extends DNNEstimator {

	public DNNRegressor(SavedModel savedModel, String head){
		super(savedModel, head);
	}

	@Override
	public NeuralNetwork encodeModel(TensorFlowEncoder encoder){
		DataField dataField = encoder.createDataField(FieldName.create("_target"), OpType.CONTINUOUS, DataType.FLOAT);

		NeuralNetwork neuralNetwork = encodeNeuralNetwork(encoder);

		List<NeuralLayer> neuralLayers = neuralNetwork.getNeuralLayers();

		NeuralLayer neuralLayer = Iterables.getLast(neuralLayers);

		neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

		List<Neuron> neurons = neuralLayer.getNeurons();

		ContinuousLabel continuousLabel = new ContinuousLabel(dataField);

		neuralNetwork
			.setMiningFunction(MiningFunction.REGRESSION)
			.setMiningSchema(ModelUtil.createMiningSchema(continuousLabel))
			.setNeuralOutputs(NeuralNetworkUtil.createRegressionNeuralOutputs(neurons, continuousLabel));

		return neuralNetwork;
	}

	public static final String REGRESSION_HEAD = "dnn/regression_head/predictions/scores";
}