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

import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

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
	public double floatToDouble(float value){
		return Double.parseDouble(Float.toString(value));
	}
}