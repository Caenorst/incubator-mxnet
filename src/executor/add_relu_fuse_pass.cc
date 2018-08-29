/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2016 by Contributors
 * \file add_relu_fuse_pass.cc
 * \brief detect whether fused add + relu is possible
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include "./exec_pass.h"

namespace mxnet {
namespace exec {

Graph FuseAddRelu(Graph&& g) {
  static const Op* add_op = Op::Get("elemwise_add");
  static const Op* activation_op = Op::Get("Activation");
  static const Op* add_relu_op = nnvm::Op::Get("add_relu");

  // to_delete_relu will contains all pointes to Relu nodes that will be fused
  // they will be deleted automatically by shared_ptr once disconnected
  std::set<nnvm::Node*> to_delete_relu = {};

  DFSVisit(g.outputs, [&to_delete_relu](const nnvm::NodePtr &node) {
    if (node->op() == activation_op &&
        node->attrs.dict.at("act_type") == "relu" &&
        node->inputs[0].node->op() == add_op &&
        node->inputs[0].node.unique()) { // add output have to be used only by the relu
      node->inputs[0].node->attrs.name = "add_" + node->attrs.name;
      node->inputs[0].node->attrs.op = add_relu_op;
      to_delete_relu.insert(node.get());
    }
  });
  DFSVisit(g.outputs, [&to_delete_relu](const nnvm::NodePtr &node) {
    for (auto& e : node->inputs) {
      nnvm::Node* hash = e.node.get();
      if (to_delete_relu.count(hash) != 0) {
        e = e.node->inputs[0];
      }
    }
  });
  Graph new_g;
  new_g.outputs = g.outputs;
  return new_g;
}

}  // namespace exec
}  // namespace mxnet
