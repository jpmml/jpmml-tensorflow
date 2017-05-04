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

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

import org.tensorflow.Tensor;

public class TensorUtil {

	private TensorUtil(){
	}

	static
	public float toFloatScalar(Tensor tensor){

		try {
			return tensor.floatValue();
		} catch(Exception e){
			float[] values = toFloatArray(tensor);

			if(values.length != 1){
				throw new IllegalArgumentException("Expected 1-element array, got " + Arrays.toString(values));
			}

			return values[0];
		}
	}

	static
	public float[] toFloatArray(Tensor tensor){
		FloatBuffer floatBuffer = FloatBuffer.allocate(tensor.numElements());

		tensor.writeTo(floatBuffer);

		return floatBuffer.array();
	}

	static
	public double[] toDoubleArray(Tensor tensor){
		DoubleBuffer doubleBuffer = DoubleBuffer.allocate(tensor.numElements());

		tensor.writeTo(doubleBuffer);

		return doubleBuffer.array();
	}

	static
	public int[] toIntArray(Tensor tensor){
		IntBuffer intBuffer = IntBuffer.allocate(tensor.numElements());

		tensor.writeTo(intBuffer);

		return intBuffer.array();
	}
}