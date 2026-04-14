/*
 * Copyright 2026 CloudWeGo Authors
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

// Package permission provides a ChatModelAgentMiddleware that gates tool execution
// behind a user-defined permission check (BeforeToolCall). It supports three decisions:
// Allow (execute the tool), Deny (return a deny message as tool result), and Ask
// (interrupt the agent loop via StatefulInterrupt for external approval).
package permission

import (
	"context"
	"fmt"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

func init() {
	schema.RegisterName[*AskInfo]("_eino_adk_permission_ask_info")
	schema.RegisterName[*AskState]("_eino_adk_permission_ask_state")
}

// Decision represents the outcome of a permission check.
type Decision string

const (
	// Allow permits the tool to execute.
	Allow Decision = "allow"
	// Deny blocks the tool and returns a deny message as the tool result.
	Deny Decision = "deny"
	// Ask interrupts the agent loop to await external approval.
	Ask Decision = "ask"
)

// ToolCallDecision is the result of a BeforeToolCall evaluation.
type ToolCallDecision struct {
	Decision     Decision
	Message      string
	UpdatedInput string
	Reason       string
}

// BeforeToolCall is the user-provided evaluation function invoked before each tool call.
// It returns a ToolCallDecision that determines whether the call is allowed, denied, or
// requires interactive approval. Returning an error signals an infrastructure failure
// and aborts the agent loop; permission denials should use Decision: Deny instead.
type BeforeToolCall func(ctx context.Context, toolName string, argumentsInJSON string) (*ToolCallDecision, error)

// AskInfo is the interrupt info exposed to external consumers (UI / OnAgentEvents).
// All fields are basic types to satisfy gob serialization requirements.
type AskInfo struct {
	ToolName  string
	CallID    string
	Arguments string
	Message   string
}

// AskState is the interrupt state persisted via CheckPointStore (gob serialization).
type AskState struct {
	Info *AskInfo
}

// ResumeResponse is the decision injected externally via ResumeWithParams.
type ResumeResponse struct {
	Approved     bool
	UpdatedInput string
	DenyMessage  string
}

// Middleware is a ChatModelAgentMiddleware that gates tool execution behind
// a user-defined BeforeToolCall permission check.
type Middleware struct {
	*adk.BaseChatModelAgentMiddleware
	beforeToolCall BeforeToolCall
}

// NewMiddleware creates a new permission Middleware with the given BeforeToolCall evaluator.
func NewMiddleware(beforeToolCall BeforeToolCall) *Middleware {
	return &Middleware{
		BaseChatModelAgentMiddleware: &adk.BaseChatModelAgentMiddleware{},
		beforeToolCall:               beforeToolCall,
	}
}

type gateResult struct {
	allowed      bool
	denyResult   string
	updatedInput string
}

func (m *Middleware) permissionGate(
	ctx context.Context,
	tCtx *adk.ToolContext,
	argumentsInJSON string,
) (*gateResult, error) {
	wasInterrupted, _, savedState := tool.GetInterruptState[*AskState](ctx)
	isTarget, hasData, response := tool.GetResumeContext[*ResumeResponse](ctx)

	if wasInterrupted && !isTarget {
		return nil, tool.StatefulInterrupt(ctx, savedState.Info, savedState)
	}

	if isTarget && hasData {
		if !response.Approved {
			return &gateResult{denyResult: formatDenyResult(tCtx.Name, response.DenyMessage)}, nil
		}
		input := argumentsInJSON
		if response.UpdatedInput != "" {
			input = response.UpdatedInput
		}
		return &gateResult{allowed: true, updatedInput: input}, nil
	}

	if isTarget && !hasData {
		return nil, fmt.Errorf("permission: tool %s was targeted for resume but received nil or type-mismatched ResumeResponse", tCtx.Name)
	}

	decision, err := m.beforeToolCall(ctx, tCtx.Name, argumentsInJSON)
	if err != nil {
		return nil, fmt.Errorf("permission check failed: %w", err)
	}
	if decision == nil {
		return nil, fmt.Errorf("permission: BeforeToolCall returned nil decision for tool %s", tCtx.Name)
	}

	switch decision.Decision {
	case Allow:
		input := argumentsInJSON
		if decision.UpdatedInput != "" {
			input = decision.UpdatedInput
		}
		return &gateResult{allowed: true, updatedInput: input}, nil

	case Deny:
		return &gateResult{denyResult: formatDenyResult(tCtx.Name, decision.Message)}, nil

	case Ask:
		info := &AskInfo{
			ToolName:  tCtx.Name,
			CallID:    tCtx.CallID,
			Arguments: argumentsInJSON,
			Message:   decision.Message,
		}
		state := &AskState{Info: info}
		return nil, tool.StatefulInterrupt(ctx, info, state)

	default:
		return &gateResult{denyResult: formatDenyResult(tCtx.Name,
			fmt.Sprintf("unknown permission decision %q", decision.Decision))}, nil
	}
}

// WrapInvokableToolCall intercepts synchronous tool calls with a permission gate.
func (m *Middleware) WrapInvokableToolCall(
	ctx context.Context,
	endpoint adk.InvokableToolCallEndpoint,
	tCtx *adk.ToolContext,
) (adk.InvokableToolCallEndpoint, error) {
	return func(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
		result, err := m.permissionGate(ctx, tCtx, argumentsInJSON)
		if err != nil {
			return "", err
		}
		if !result.allowed {
			return result.denyResult, nil
		}
		return endpoint(ctx, result.updatedInput, opts...)
	}, nil
}

// WrapStreamableToolCall intercepts streaming tool calls with a permission gate.
func (m *Middleware) WrapStreamableToolCall(
	ctx context.Context,
	endpoint adk.StreamableToolCallEndpoint,
	tCtx *adk.ToolContext,
) (adk.StreamableToolCallEndpoint, error) {
	return func(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (*schema.StreamReader[string], error) {
		result, err := m.permissionGate(ctx, tCtx, argumentsInJSON)
		if err != nil {
			return nil, err
		}
		if !result.allowed {
			sr, sw := schema.Pipe[string](1)
			sw.Send(result.denyResult, nil)
			sw.Close()
			return sr, nil
		}
		return endpoint(ctx, result.updatedInput, opts...)
	}, nil
}

func formatDenyResult(toolName, message string) string {
	return fmt.Sprintf("Permission denied for tool %s: %s", toolName, message)
}
