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

import com.google.common.collect.Iterables;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.OpType;
import org.dmg.pmml.neural_network.Connection;
import org.dmg.pmml.neural_network.NeuralLayer;
import org.dmg.pmml.neural_network.NeuralNetwork;
import org.dmg.pmml.neural_network.Neuron;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.neural_network.NeuralNetworkUtil;

public class DNNClassifier extends DNNEstimator {

	public DNNClassifier(SavedModel savedModel, String head){
		super(savedModel, head);
	}

	@Override
	public NeuralNetwork encodeModel(TensorFlowEncoder encoder){
		DataField dataField = encoder.createDataField(FieldName.create("_target"), OpType.CATEGORICAL, DataType.INTEGER);

		NeuralNetwork neuralNetwork = encodeNeuralNetwork(encoder);

		List<NeuralLayer> neuralLayers = neuralNetwork.getNeuralLayers();

		NeuralLayer neuralLayer = Iterables.getLast(neuralLayers);

		List<Neuron> neurons = neuralLayer.getNeurons();

		List<String> categories;

		if(neurons.size() == 1){
			neuralLayer.setActivationFunction(NeuralNetwork.ActivationFunction.LOGISTIC);

			Neuron neuron = Iterables.getOnlyElement(neurons);

			neuralLayer = new NeuralLayer()
				.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY);

			categories = Arrays.asList("0", "1");

			// p(no event) = 1 - p(event)
			Neuron passiveNeuron = new Neuron()
				.setId(String.valueOf(neuralLayers.size() + 1) + "/" + categories.get(0))
				.setBias(floatToDouble(1f))
				.addConnections(new Connection(neuron.getId(), -1f));

			// p(event)
			Neuron activeNeuron = new Neuron()
				.setId(String.valueOf(neuralLayers.size() + 1) + "/" + categories.get(1))
				.setBias(null)
				.addConnections(new Connection(neuron.getId(), 1f));

			neuralLayer.addNeurons(passiveNeuron, activeNeuron);

			neuralNetwork.addNeuralLayers(neuralLayer);

			neurons = neuralLayer.getNeurons();
		} else

		if(neurons.size() > 2){
			neuralLayer
				.setActivationFunction(NeuralNetwork.ActivationFunction.IDENTITY)
				.setNormalizationMethod(NeuralNetwork.NormalizationMethod.SOFTMAX);

			categories = new ArrayList<>();

			for(int i = 0; i < neurons.size(); i++){
				String category = String.valueOf(i);

				categories.add(category);
			}
		} else

		{
			throw new IllegalArgumentException();
		}

		dataField = encoder.toCategorical(dataField.getName(), categories);

		CategoricalLabel categoricalLabel = new CategoricalLabel(dataField);

		neuralNetwork
			.setMiningFunction(MiningFunction.CLASSIFICATION)
			.setMiningSchema(ModelUtil.createMiningSchema(categoricalLabel))
			.setNeuralOutputs(NeuralNetworkUtil.createClassificationNeuralOutputs(neurons, categoricalLabel))
			.setOutput(ModelUtil.createProbabilityOutput(categoricalLabel));

		return neuralNetwork;
	}

	public static final String BINARY_LOGISTIC_HEAD = "dnn/binary_logistic_head/predictions/probabilities";
	public static final String MULTI_CLASS_HEAD = "dnn/multi_class_head/predictions/probabilities";
}