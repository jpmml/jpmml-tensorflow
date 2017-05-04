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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Iterables;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.NodeDef;

public class SavedModel implements AutoCloseable {

	private SavedModelBundle bundle = null;

	private MetaGraphDef metaGraphDef = null;

	private Map<String, NodeDef> nodeMap = null;


	public SavedModel(SavedModelBundle bundle) throws InvalidProtocolBufferException {
		setBundle(bundle);

		byte[] metaGraphDefBytes = bundle.metaGraphDef();

		MetaGraphDef metaGraphDef = MetaGraphDef.parseFrom(metaGraphDefBytes);

		setMetaGraphDef(metaGraphDef);

		GraphDef graphDef = metaGraphDef.getGraphDef();

		Map<String, NodeDef> nodeMap = new LinkedHashMap<>();

		List<NodeDef> nodeDefs = graphDef.getNodeList();
		for(NodeDef nodeDef : nodeDefs){
			nodeMap.put(nodeDef.getName(), nodeDef);
		}

		setNodeMap(nodeMap);
	}

	@Override
	public void close(){
		SavedModelBundle bundle = getBundle();

		bundle.close();
	}

	public Tensor run(String name){
		Session session = getSession();

		Runner runner = (session.runner()).fetch(name);

		List<Tensor> tensors = runner.run();

		return Iterables.getOnlyElement(tensors);
	}

	public Operation getOperation(String name){
		Graph graph = getGraph();

		return graph.operation(name);
	}

	public NodeDef getNodeDef(String name){
		Map<String, NodeDef> nodeMap = getNodeMap();

		NodeDef nodeDef = nodeMap.get(name);
		if(nodeDef == null){
			throw new IllegalArgumentException(name);
		}

		return nodeDef;
	}

	public Session getSession(){
		SavedModelBundle bundle = getBundle();

		return bundle.session();
	}

	public Graph getGraph(){
		SavedModelBundle bundle = getBundle();

		return bundle.graph();
	}

	public SavedModelBundle getBundle(){
		return this.bundle;
	}

	private void setBundle(SavedModelBundle bundle){
		this.bundle = bundle;
	}

	public MetaGraphDef getMetaGraphDef(){
		return this.metaGraphDef;
	}

	private void setMetaGraphDef(MetaGraphDef metaGraphDef){
		this.metaGraphDef = metaGraphDef;
	}

	public Map<String, NodeDef> getNodeMap(){
		return this.nodeMap;
	}

	private void setNodeMap(Map<String, NodeDef> nodeMap){
		this.nodeMap = nodeMap;
	}
}