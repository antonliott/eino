// Copyright 2024 CloudWeGo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package compose provides primitives for building and executing AI pipelines
// as directed acyclic graphs (DAGs).
package compose

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// NodeType represents the type of a node in the graph.
type NodeType string

const (
	// NodeTypeLLM represents a large language model node.
	NodeTypeLLM NodeType = "llm"
	// NodeTypeTool represents a tool/function node.
	NodeTypeTool NodeType = "tool"
	// NodeTypeRetriever represents a retriever node.
	NodeTypeRetriever NodeType = "retriever"
	// NodeTypeTransform represents a data transformation node.
	NodeTypeTransform NodeType = "transform"
)

// Node represents a single processing unit in the graph.
type Node struct {
	ID       string
	Type     NodeType
	Handler  NodeHandler
	edges    []string // IDs of successor nodes
}

// NodeHandler defines the function signature for node processing logic.
type NodeHandler func(ctx context.Context, input map[string]any) (map[string]any, error)

// Graph is a directed acyclic graph used to compose AI pipeline components.
type Graph struct {
	mu    sync.RWMutex
	nodes map[string]*Node
	start string
	end   string
}

// NewGraph creates a new empty Graph.
func NewGraph() *Graph {
	return &Graph{
		nodes: make(map[string]*Node),
	}
}

// AddNode adds a node with the given id, type, and handler to the graph.
// Returns an error if a node with the same id already exists.
func (g *Graph) AddNode(id string, nodeType NodeType, handler NodeHandler) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.nodes[id]; exists {
		return fmt.Errorf("node with id %q already exists", id)
	}
	g.nodes[id] = &Node{
		ID:      id,
		Type:    nodeType,
		Handler: handler,
	}
	return nil
}

// AddEdge adds a directed edge from node `from` to node `to`.
// Returns an error if either node does not exist.
func (g *Graph) AddEdge(from, to string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	fromNode, ok := g.nodes[from]
	if !ok {
		return fmt.Errorf("source node %q not found", from)
	}
	if _, ok := g.nodes[to]; !ok {
		return fmt.Errorf("destination node %q not found", to)
	}
	fromNode.edges = append(fromNode.edges, to)
	return nil
}

// SetEntryPoint sets the starting node for graph execution.
func (g *Graph) SetEntryPoint(id string) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, ok := g.nodes[id]; !ok {
		return fmt.Errorf("entry point node %q not found", id)
	}
	g.start = id
	return nil
}

// SetFinishPoint sets the terminal node for graph execution.
func (g *Graph) SetFinishPoint(id string) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, ok := g.nodes[id]; !ok {
		return fmt.Errorf("finish point node %q not found", id)
	}
	g.end = id
	return nil
}

// Run executes the graph starting from the entry point with the provided input.
// Nodes are executed sequentially following the defined edges.
func (g *Graph) Run(ctx context.Context, input map[string]any) (map[string]any, error) {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.start == "" {
		return nil, errors.New("entry point not set")
	}

	current := g.start
	state := input

	for {
		node, ok := g.nodes[current]
		if !ok {
			return nil, fmt.Errorf("node %q not found during execution", current)
		}

		var err error
		state, err = node.Handler(ctx, state)
		if err != nil {
			return nil, fmt.Errorf("node %q failed: %w", current, err)
		}

		if current == g.end || len(node.edges) == 0 {
			break
		}
		// Follow the first edge (linear execution); branching support is future work.
		current = node.edges[0]
	}

	return state, nil
}
