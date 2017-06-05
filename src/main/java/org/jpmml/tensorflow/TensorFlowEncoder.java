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

import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.ModelEncoder;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.framework.NodeDef;

public class TensorFlowEncoder extends ModelEncoder {

	public DataField ensureDataField(SavedModel savedModel, NodeDef placeholder){

		if(!("Placeholder").equals(placeholder.getOp())){
			throw new IllegalArgumentException(placeholder.getName());
		}

		FieldName name = FieldName.create(placeholder.getName());

		DataField dataField = getDataField(name);
		if(dataField == null){
			Operation operation = savedModel.getOperation(placeholder.getName());

			Output output = operation.output(0);

			dataField = createDataField(name, TypeUtil.getOpType(output), TypeUtil.getDataType(output));
		}

		return dataField;
	}

	public DataField ensureContinuousDataField(SavedModel savedModel, NodeDef placeholder){
		DataField dataField = ensureDataField(savedModel, placeholder);

		return toContinuous(dataField.getName());
	}

	public DataField ensureCategoricalDataField(SavedModel savedModel, NodeDef placeholder, List<String> values){
		DataField dataField = ensureDataField(savedModel, placeholder);

		return toCategorical(dataField.getName(), values);
	}

	public ContinuousFeature createContinuousFeature(SavedModel savedModel, NodeDef placeholder){
		NodeDef cast = null;

		if(("Cast").equals(placeholder.getOp())){
			cast = placeholder;
			placeholder = savedModel.getNodeDef(placeholder.getInput(0));
		}

		DataField dataField = ensureContinuousDataField(savedModel, placeholder);

		ContinuousFeature result = new ContinuousFeature(this, dataField);

		if(cast != null){
			Operation operation = savedModel.getOperation(cast.getName());

			Output output = operation.output(0);

			result = result.toContinuousFeature(TypeUtil.getDataType(output));
		}

		return result;
	}

	public List<BinaryFeature> createBinaryFeatures(SavedModel savedModel, NodeDef placeholder, List<String> categories){
		DataField dataField = ensureCategoricalDataField(savedModel, placeholder, categories);

		List<BinaryFeature> result = new ArrayList<>();

		for(String category : categories){
			BinaryFeature binaryFeature = new BinaryFeature(this, dataField, category);

			result.add(binaryFeature);
		}

		return result;
	}
}