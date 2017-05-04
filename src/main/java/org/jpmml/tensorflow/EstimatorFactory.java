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

import java.util.Map;

import org.tensorflow.framework.NodeDef;

public class EstimatorFactory {

	protected EstimatorFactory(){
	}

	public Estimator newEstimator(SavedModel savedModel){
		Map<String, NodeDef> nodeMap = savedModel.getNodeMap();

		if(nodeMap.containsKey(LinearRegressor.HEAD)){
			return new LinearRegressor(savedModel);
		} else

		{
			throw new IllegalArgumentException();
		}
	}

	static
	public EstimatorFactory newInstance(){
		return new EstimatorFactory();
	}
}