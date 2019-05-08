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
 * Copyright (c) 2019 by Contributors
 * \file simple_partition_pass.h
 * \brief 
 * \author Clement Fuji Tsang
 */
#ifndef MXNET_OPERATOR_SUBGRAPH_SIMPLE_PARTITION_GRAPH_PASS_H_
#define MXNET_OPERATOR_SUBGRAPH_SIMPLE_PARTITION_GRAPH_PASS_H_

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <nnvm/graph_attr_types.h>

namespace mxnet {
namespace op {

/*!
 * \brief Custom graph class, which will contain bi-directional nodes
 * we need to compute DFS and reverse DFS for graph partitioning
 */
class BidirectionalGraph {
 public:
  struct Node {
    nnvm::Node* nnvmptr;
    std::vector<Node*> inputs;
    std::vector<Node*> outputs;
  };
  
  using SubgraphsMap =
      std::unordered_map<const nnvm::Node*, std::shared_ptr<std::unordered_set<const nnvm::Node*> > >;
  
  explicit BidirectionalGraph(const nnvm::Graph &g) {
    auto& idx = g.indexed_graph();
    auto num_nodes = idx.num_nodes();
    nodes.reserve(num_nodes);
    nnvm2nid.reserve(num_nodes);
    outputs.reserve(idx.outputs().size());
    nnvm::DFSVisit(g.outputs, [this](const nnvm::NodePtr& n) {
      Node new_node;
      new_node.nnvmptr = n.get();
      nnvm2nid[n.get()] = static_cast<uint32_t>(nodes.size());
      nodes.emplace_back(std::move(new_node));
    });
    for (const auto& it : nnvm2nid) {
      nnvm::Node* nnvmnode = it.first;
      uint32_t nid = it.second;
      for (auto& n : nnvmnode->inputs) {
        uint32_t input_nid = nnvm2nid[n.node.get()];
        nodes[input_nid].outputs.emplace_back(&nodes[nid]);
        nodes[nid].inputs.emplace_back(&nodes[input_nid]);
      }
    }
    for (auto& e : g.outputs) {
      uint32_t nid = nnvm2nid[e.node.get()];
      outputs.emplace_back(&nodes[nid]);
    }
  }

  template<typename FCompatible>
  std::unique_ptr<SubgraphsMap> get_subsets(FCompatible is_compatible) {
    auto subgraphs_map = std::make_unique<SubgraphsMap>();
    subgraphs_map->reserve(nodes.size());
    std::unordered_set<Node*> incomp_set;
    std::unordered_set<Node*> all_set(nodes.size());
    std::vector<PairSet> separation_sets;
    for (Node& node : nodes) {
      if (!is_compatible(node.nnvmptr)) {
        incomp_set.insert(&node);
        std::unordered_set<Node*> in_graph;
        std::unordered_set<Node*> out_graph;
        std::vector<Node*> dummy_head;
        dummy_head.emplace_back(&node);
        DFS(dummy_head, false, [&out_graph](Node* node) {
          out_graph.insert(node);
        });
        DFS(dummy_head, true, [&in_graph](Node* node) {
          in_graph.insert(node);
        });
        if (!(in_graph.empty() || out_graph.empty()))
          separation_sets.push_back(std::make_pair(in_graph, out_graph));
      }
      all_set.emplace(&node);
    }
    IncompMap incomp_map;
    std::unordered_set<Node*> comp_set;
    comp_set.insert(all_set.begin(), all_set.end());
    for (Node* n : incomp_set) {
      comp_set.erase(n);
    }
    for (Node* n : comp_set) {
      for (PairSet p : separation_sets) {
        if (p.first.count(n)) {
          incomp_map[n].insert(p.second.begin(), p.second.end());
        } else if (p.second.count(n)) {
          incomp_map[n].insert(p.first.begin(), p.first.end());
        }
      }
      for (Node* incomp_n : incomp_set) {
        incomp_map[n].erase(incomp_n);
      }
    }
    std::unordered_set<Node*> unused_set;
    unused_set.reserve(comp_set.size());
    for (auto& n : comp_set) {
      unused_set.insert(n);
    }
    std::unordered_set<Node*> visited;
    std::deque<Node*> stack(outputs.begin(), outputs.end());
    while (!stack.empty()) {
      Node* vertex = stack.front();
      stack.pop_front();
      if (!visited.count(vertex)) {
        visited.insert(vertex);
        if (unused_set.count(vertex)) {
          auto subgraph_vec = naive_grow_subgraph(vertex, &unused_set, &incomp_map);
          auto subgraph_set = std::make_shared<std::unordered_set<const nnvm::Node*> >(
                                             subgraph_vec.begin(), subgraph_vec.end());
          for (auto subgraph_node : subgraph_vec) {
            subgraphs_map->insert({subgraph_node, subgraph_set});
          } 
        }
        for (Node* input : vertex->inputs) {
          stack.emplace_back(input);
        }
      }
    }
    return subgraphs_map;
  }

 private:
  using PairSet = std::pair<std::unordered_set<Node*>, std::unordered_set<Node*>>;
  using PairVec = std::pair<std::vector<Node*>, std::vector<Node*>>;
  using IncompMap = std::unordered_map<Node*, std::unordered_set<Node*>>;

 template <typename FVisit>
  void DFS(const std::vector<Node*>& heads, bool reverse, FVisit fvisit) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> vec(heads.begin(), heads.end());
    visited.reserve(heads.size());
    while (!vec.empty()) {
      Node* vertex = vec.back();
      vec.pop_back();
      if (visited.count(vertex) == 0) {
        visited.insert(vertex);
        fvisit(vertex);
        std::vector<Node*> nexts = reverse ? vertex->inputs : vertex->outputs;
        for (Node* node : nexts) {
          if (visited.count(node) == 0) {
            vec.emplace_back(node);
          }
        }
      }
    }
  }

  std::vector<nnvm::Node*> naive_grow_subgraph(Node* head,
                                               std::unordered_set<Node*>* unused_set,
                                               IncompMap* incomp_map) {
    std::vector<nnvm::Node*> subgraph_vec;
    std::unordered_set<Node*> incomp_set;
    std::deque<Node*> stack;
    stack.emplace_back(head);
    while (!stack.empty()) {
      Node* vertex = stack.back();
      stack.pop_back();
      if (unused_set->count(vertex) && !incomp_set.count(vertex)) {
        unused_set->erase(vertex);
        subgraph_vec.push_back(vertex->nnvmptr);
        incomp_set.insert((*incomp_map)[vertex].begin(), (*incomp_map)[vertex].end());
        for (Node* input : vertex->inputs) {
          if (unused_set->count(input) && !incomp_set.count(input)) {
            stack.emplace_back(input);
          }
        }
        for (Node* output : vertex->outputs) {
          if (unused_set->count(output) && !incomp_set.count(output)) {
            stack.emplace_back(output);
          }
        }
      }
    }
    return subgraph_vec;
  }

  std::vector<Node> nodes;
  std::unordered_map<nnvm::Node*, uint32_t> nnvm2nid;
  std::vector<Node*> outputs;

};  // class BidirectionalGraph

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_SIMPLE_PARTITION_GRAPH_PASS_H_
