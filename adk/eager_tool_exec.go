/*
 * Copyright 2025 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package adk

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"sync"

	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

type eagerToolResult struct {
	output         string
	enhancedOutput *schema.ToolResult
	useEnhanced    bool
	err            error
}

type eagerCoord struct {
	mu      sync.Mutex
	results map[string]*eagerToolResult
	doneCh  chan struct{}
	once    sync.Once
	aborted bool
}

func newEagerCoord() *eagerCoord {
	return &eagerCoord{
		results: make(map[string]*eagerToolResult),
		doneCh:  make(chan struct{}),
	}
}

func (c *eagerCoord) abort() {
	c.mu.Lock()
	c.aborted = true
	c.mu.Unlock()
}

func (c *eagerCoord) isAborted() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.aborted
}

func (c *eagerCoord) storeResult(callID string, result *eagerToolResult) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.results[callID] = result
}

func (c *eagerCoord) markDone() {
	c.once.Do(func() {
		close(c.doneCh)
	})
}

func (c *eagerCoord) waitDone(ctx context.Context) {
	select {
	case <-c.doneCh:
	case <-ctx.Done():
	}
}

func (c *eagerCoord) collectResults() (
	executedTools map[string]string,
	executedEnhancedTools map[string]*schema.ToolResult,
	failedToolCallIDs []string,
	err error,
) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.aborted {
		return nil, nil, nil, nil
	}

	for callID, result := range c.results {
		if result.err != nil {
			failedToolCallIDs = append(failedToolCallIDs, callID)
			continue
		}
		if result.useEnhanced {
			if executedEnhancedTools == nil {
				executedEnhancedTools = make(map[string]*schema.ToolResult)
			}
			executedEnhancedTools[callID] = result.enhancedOutput
		} else {
			if executedTools == nil {
				executedTools = make(map[string]string)
			}
			executedTools[callID] = result.output
		}
	}

	return executedTools, executedEnhancedTools, failedToolCallIDs, nil
}

type eagerCoordHolder struct {
	mu    sync.Mutex
	coord *eagerCoord
}

func (h *eagerCoordHolder) Store(c *eagerCoord) {
	h.mu.Lock()
	h.coord = c
	h.mu.Unlock()
}

func (h *eagerCoordHolder) Load() *eagerCoord {
	h.mu.Lock()
	c := h.coord
	h.mu.Unlock()
	return c
}

type eagerToolExecutorMiddleware[M MessageType] struct {
	toolsNode           *compose.ToolsNode
	toolNodeKey         string
	coordPtr            *eagerCoordHolder
	executeSequentially bool
}

func (e *eagerToolExecutorMiddleware[M]) runEager(ctx context.Context, stream *schema.StreamReader[M], coord *eagerCoord) {
	defer stream.Close()
	defer coord.markDone()

	accumulators := map[int]*toolCallAccumulator{}
	dispatched := map[string]bool{}
	var wg sync.WaitGroup

	dispatchTool := func(tc schema.ToolCall) {
		if coord.isAborted() {
			coord.storeResult(tc.ID, &eagerToolResult{err: context.Canceled})
			return
		}
		if e.executeSequentially {
			result := e.executeSingleTool(ctx, tc, coord)
			coord.storeResult(tc.ID, result)
		} else {
			wg.Add(1)
			go func() {
				defer wg.Done()
				result := e.executeSingleTool(ctx, tc, coord)
				coord.storeResult(tc.ID, result)
			}()
		}
	}

	for {
		chunk, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			coord.abort()
			wg.Wait()
			return
		}

		for _, tc := range extractToolCalls(chunk) {
			idx := derefIndex(tc.Index)
			acc := getOrCreateAccumulator(accumulators, idx)
			acc.merge(tc)

			if !dispatched[acc.id] && acc.id != "" && isArgsComplete(acc.args.String()) {
				dispatched[acc.id] = true
				dispatchTool(acc.toToolCall())
			}
		}
	}

	for _, acc := range accumulators {
		if !dispatched[acc.id] && acc.id != "" && acc.name != "" {
			dispatched[acc.id] = true
			dispatchTool(acc.toToolCall())
		}
	}

	wg.Wait()
}

func (e *eagerToolExecutorMiddleware[M]) executeSingleTool(ctx context.Context, tc schema.ToolCall, coord *eagerCoord) *eagerToolResult {
	if coord.isAborted() {
		return &eagerToolResult{err: context.Canceled}
	}

	toolNodeCtx := compose.AppendAddressSegment(ctx, compose.AddressSegmentNode, e.toolNodeKey)

	result, err := e.toolsNode.InvokeSingleToolCall(toolNodeCtx, tc)
	if err != nil {
		return &eagerToolResult{err: err}
	}

	return &eagerToolResult{
		output:         result.Output,
		enhancedOutput: result.EnhancedOutput,
		useEnhanced:    result.UseEnhanced,
	}
}

func extractToolCalls[M MessageType](msg M) []schema.ToolCall {
	switch v := any(msg).(type) {
	case *schema.Message:
		return v.ToolCalls
	case *schema.AgenticMessage:
		var tcs []schema.ToolCall
		for _, block := range v.ContentBlocks {
			if block.Type != schema.ContentBlockTypeFunctionToolCall || block.FunctionToolCall == nil {
				continue
			}
			tcs = append(tcs, schema.ToolCall{
				ID: block.FunctionToolCall.CallID,
				Function: schema.FunctionCall{
					Name:      block.FunctionToolCall.Name,
					Arguments: block.FunctionToolCall.Arguments,
				},
				Extra: block.Extra,
			})
		}
		return tcs
	}
	return nil
}

type toolCallAccumulator struct {
	id   string
	name string
	args strings.Builder
	typ  string
}

func (a *toolCallAccumulator) merge(tc schema.ToolCall) {
	if tc.ID != "" && a.id == "" {
		a.id = tc.ID
	}
	if tc.Function.Name != "" && a.name == "" {
		a.name = tc.Function.Name
	}
	if tc.Type != "" && a.typ == "" {
		a.typ = tc.Type
	}
	if tc.Function.Arguments != "" {
		a.args.WriteString(tc.Function.Arguments)
	}
}

func (a *toolCallAccumulator) toToolCall() schema.ToolCall {
	return schema.ToolCall{
		ID:   a.id,
		Type: a.typ,
		Function: schema.FunctionCall{
			Name:      a.name,
			Arguments: a.args.String(),
		},
	}
}

func getOrCreateAccumulator(m map[int]*toolCallAccumulator, idx int) *toolCallAccumulator {
	if acc, ok := m[idx]; ok {
		return acc
	}
	acc := &toolCallAccumulator{}
	m[idx] = acc
	return acc
}

func derefIndex(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func isArgsComplete(args string) bool {
	if args == "" {
		return false
	}
	args = strings.TrimSpace(args)
	if !strings.HasPrefix(args, "{") {
		return false
	}
	return json.Valid([]byte(args))
}
