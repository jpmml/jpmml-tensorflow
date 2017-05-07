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

import org.dmg.pmml.DataField;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.Model;
import org.dmg.pmml.OpType;
import org.dmg.pmml.PMML;
import org.jpmml.converter.CategoricalFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.tensorflow.Operation;
import org.tensorflow.Output;
import org.tensorflow.framework.NodeDef;

abstract
public class Estimator {

	private SavedModel savedModel = null;

	private String head = null;


	public Estimator(SavedModel savedModel, String head){
		setSavedModel(savedModel);
		setHead(head);
	}

	abstract
	public Model encodeModel(TensorFlowEncoder encoder);

	public PMML encodePMML(){
		TensorFlowEncoder encoder = new TensorFlowEncoder();

		Model model = encodeModel(encoder);

		PMML pmml = encoder.encodePMML(model);

		return pmml;
	}

	public Feature encodeContinuousFeature(NodeDef nodeDef, TensorFlowEncoder encoder){
		SavedModel savedModel = getSavedModel();

		NodeDef castNodeDef;
		NodeDef placeholderNodeDef;

		if(("Cast").equals(nodeDef.getOp())){
			castNodeDef = checkOp(nodeDef, "Cast");
			placeholderNodeDef = checkOp(savedModel.getNodeDef(nodeDef.getInput(0)), "Placeholder");
		} else

		{
			castNodeDef = null;
			placeholderNodeDef = checkOp(nodeDef, "Placeholder");
		}

		Operation placeholderOperation = savedModel.getOperation(placeholderNodeDef.getName());
		Output placeholderOutput = placeholderOperation.output(0);

		DataField dataField = encoder.createDataField(FieldName.create(placeholderNodeDef.getName()), OpType.CONTINUOUS, TypeUtil.translateDataType(placeholderOutput.dataType()));

		Feature feature = new ContinuousFeature(encoder, dataField);

		if(castNodeDef != null){
			Operation castOperation = savedModel.getOperation(castNodeDef.getName());
			Output castOutput = castOperation.output(0);

			feature = feature.toContinuousFeature(TypeUtil.translateDataType(castOutput.dataType()));
		}

		return feature;
	}

	public Feature encodeCategoricalFeature(NodeDef nodeDef, List<String> categories, TensorFlowEncoder encoder){
		SavedModel savedModel = getSavedModel();

		NodeDef placeholderNodeDef = checkOp(nodeDef, "Placeholder");

		Operation placeholderOperation = savedModel.getOperation(placeholderNodeDef.getName());
		Output placeholderOutput = placeholderOperation.output(0);

		DataField dataField = encoder.createDataField(FieldName.create(placeholderNodeDef.getName()), OpType.CATEGORICAL, TypeUtil.translateDataType(placeholderOutput.dataType()), categories);

		Feature feature = new CategoricalFeature(encoder, dataField);

		return feature;
	}

	public SavedModel getSavedModel(){
		return this.savedModel;
	}

	private void setSavedModel(SavedModel savedModel){
		this.savedModel = savedModel;
	}

	public String getHead(){
		return this.head;
	}

	private void setHead(String head){
		this.head = head;
	}

	static
	protected NodeDef checkOp(NodeDef nodeDef, String op){

		if(!(op).equals(nodeDef.getOp())){
			throw new IllegalArgumentException("Expected " + op + ", got " + nodeDef.getOp());
		}

		return nodeDef;
	}
}