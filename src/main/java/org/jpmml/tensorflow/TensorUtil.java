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

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.List;

import com.google.common.primitives.Booleans;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;

public class TensorUtil {

	private TensorUtil(){
	}

	static
	public List<?> getValues(Tensor tensor){
		DataType dataType = tensor.dataType();

		switch(dataType){
			case FLOAT:
				return Floats.asList(TensorUtil.toFloatArray(tensor));
			case DOUBLE:
				return Doubles.asList(TensorUtil.toDoubleArray(tensor));
			case INT32:
				return Ints.asList(TensorUtil.toIntArray(tensor));
			case INT64:
				return Longs.asList(TensorUtil.toLongArray(tensor));
			case STRING:
				return Arrays.asList(TensorUtil.toStringArray(tensor));
			case BOOL:
				return Booleans.asList(TensorUtil.toBooleanArray(tensor));
			default:
				throw new IllegalArgumentException();
		}
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

	static
	public long[] toLongArray(Tensor tensor){
		LongBuffer longBuffer = LongBuffer.allocate(tensor.numElements());

		tensor.writeTo(longBuffer);

		return longBuffer.array();
	}

	static
	public String[] toStringArray(Tensor tensor){
		ByteBuffer byteBuffer = ByteBuffer.allocate(tensor.numBytes());

		tensor.writeTo(byteBuffer);

		byteBuffer.position(tensor.numElements() * 8);

		String[] result = new String[tensor.numElements()];

		for(int i = 0; i < result.length; i++){
			int length = byteBuffer.get();

			byte[] buffer = new byte[length];

			byteBuffer.get(buffer);

			result[i] = new String(buffer);
		}

		return result;
	}

	static
	public boolean[] toBooleanArray(Tensor tensor){
		ByteBuffer byteBuffer = ByteBuffer.allocate(tensor.numElements());

		tensor.writeTo(byteBuffer);

		boolean[] result = new boolean[tensor.numElements()];

		for(int i = 0; i < result.length; i++){
			result[i] = (byteBuffer.get(i) != 0);
		}

		return result;
	}
}