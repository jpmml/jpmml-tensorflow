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

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.framework.CollectionDef;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.NodeDef;

public class SavedModel implements AutoCloseable {

	private SavedModelBundle bundle = null;

	private MetaGraphDef metaGraphDef = null;

	private Map<String, NodeDef> nodeMap = null;

	private Map<String, Map<?, ?>> tableMap = new LinkedHashMap<>();


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

		initializeTables();
	}

	private void initializeTables(){
		Collection<String> tableInitializerNames = Collections.emptyList();

		try {
			CollectionDef collectionDef = getCollectionDef("table_initializer");

			CollectionDef.NodeList nodeList = collectionDef.getNodeList();

			tableInitializerNames = nodeList.getValueList();
		} catch(IllegalArgumentException iae){
			// Ignored
		}

		for(String tableInitializerName : tableInitializerNames){
			NodeDef tableInitializer = getNodeDef(tableInitializerName);

			String name = tableInitializer.getInput(0);

			List<?> keys;
			List<?> values;

			try(Tensor tensor = run(tableInitializer.getInput(1))){
				keys = TensorUtil.getValues(tensor);
			} // End try

			try(Tensor tensor = run(tableInitializer.getInput(2))){
				values = TensorUtil.getValues(tensor);
			}

			Map<Object, Object> table = new LinkedHashMap<>();

			if(keys.size() != values.size()){
				throw new IllegalArgumentException();
			}

			for(int i = 0; i < keys.size(); i++){
				table.put(keys.get(i), values.get(i));
			}

			putTable(name, table);
		}
	}

	@Override
	public void close(){
		SavedModelBundle bundle = getBundle();

		bundle.close();
	}

	public Tensor run(String name){
		Session session = getSession();

		Runner runner = (session.runner()).fetch(name);

		List<? extends Tensor> tensors = runner.run();

		return Iterables.getOnlyElement(tensors);
	}

	public Operation getOperation(String name){
		Graph graph = getGraph();

		return graph.operation(name);
	}

	public NodeDef getNodeDef(String name){
		Map<String, NodeDef> nodeMap = getNodeMap();

		int colon = name.indexOf(':');

		NodeDef nodeDef = nodeMap.get(colon > -1 ? name.substring(0, colon) : name);
		if(nodeDef == null){
			throw new IllegalArgumentException(name);
		}

		return nodeDef;
	}

	public CollectionDef getCollectionDef(String key){
		MetaGraphDef metaGraphDef = getMetaGraphDef();

		Map<String, CollectionDef> collectionMap = metaGraphDef.getCollectionDefMap();

		CollectionDef collectionDef = collectionMap.get(key);
		if(collectionDef == null){
			throw new IllegalArgumentException(key);
		}

		return collectionDef;
	}

	public NodeDef getOnlyInput(String name, String... ops){
		Iterable<NodeDef> inputs = getInputs(name, ops);

		return Iterables.getOnlyElement(inputs);
	}

	public Iterable<NodeDef> getInputs(String name, String... ops){
		NodeDef nodeDef = getNodeDef(name);

		Collection<Trail> trails = new LinkedHashSet<>();

		collectInputs(new ArrayDeque<>(), nodeDef, new HashSet<>(Arrays.asList(ops)), trails);

		Function<Trail, NodeDef> function = new Function<Trail, NodeDef>(){

			@Override
			public NodeDef apply(Trail trail){
				return trail.getNodeDef();
			}
		};

		Collection<NodeDef> inputs = new LinkedHashSet<>();

		Iterables.addAll(inputs, Iterables.transform(trails, function));

		return inputs;
	}

	private void collectInputs(Deque<NodeDef> parentNodeDefs, NodeDef nodeDef, Set<String> ops, Collection<Trail> trails){

		if(ops.contains(nodeDef.getOp())){
			trails.add(new Trail(parentNodeDefs, nodeDef));
		}

		List<String> inputNames = nodeDef.getInputList();
		for(String inputName : inputNames){
			NodeDef inputNodeDef = getNodeDef(inputName);

			parentNodeDefs.addFirst(inputNodeDef);

			collectInputs(parentNodeDefs, inputNodeDef, ops, trails);

			parentNodeDefs.removeFirst();
		}
	}

	public Map<?, ?> getTable(String name){
		Map<?, ?> table = this.tableMap.get(name);

		if(table == null){
			throw new IllegalArgumentException(name);
		}

		return table;
	}

	private void putTable(String name, Map<Object, Object> table){
		this.tableMap.put(name, table);
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