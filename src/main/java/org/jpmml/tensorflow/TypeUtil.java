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

import org.dmg.pmml.DataType;

public class TypeUtil {

	private TypeUtil(){
	}

	static
	public DataType translateDataType(org.tensorflow.DataType dataType){

		switch(dataType){
			case FLOAT:
				return DataType.FLOAT;
			case DOUBLE:
				return DataType.DOUBLE;
			case INT32:
			case INT64:
				return DataType.INTEGER;
			case STRING:
				return DataType.STRING;
			case BOOL:
				return DataType.BOOLEAN;
			default:
				throw new IllegalArgumentException();
		}
	}
}