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
// behind a user-defined permission check (Checker). It supports three decisions:
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

type Decision string

const (
	Allow Decision = "allow"
	Deny  Decision = "deny"
	Ask   Decision = "ask"
)

type ToolCallDecision struct {
	Decision     Decision
	Message      string
	UpdatedInput string
	Reason       string
}

// Checker is the user-provided evaluation function invoked before each tool call.
// It receives the full ToolContext (including tool name and call ID) along with
// the tool arguments as a *schema.ToolArgument, and returns a ToolCallDecision
// that determines whether the call is allowed, denied, or requires interactive
// approval. Using *schema.ToolArgument instead of a raw string ensures
// forward-compatibility when the struct gains additional fields (e.g. multimodal
// content). Returning an error signals an infrastructure failure and aborts the
// agent loop; permission denials should use Decision: Deny instead.
type Checker func(ctx context.Context, tCtx *adk.ToolContext, args *schema.ToolArgument) (*ToolCallDecision, error)

type AskInfo struct {
	ToolName  string
	CallID    string
	Arguments string
	Message   string
}

type AskState struct {
	Info *AskInfo
}

type ResumeResponse struct {
	Approved     bool
	UpdatedInput string
	DenyMessage  string
}

type Middleware struct {
	*adk.BaseChatModelAgentMiddleware
	checker Checker
}

// New creates a permission Middleware with the given Checker evaluator.
func New(checker Checker) *Middleware {
	return &Middleware{
		BaseChatModelAgentMiddleware: &adk.BaseChatModelAgentMiddleware{},
		checker:                      checker,
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
		return nil, fmt.Errorf(
			"permission: tool %q (call_id=%s) was targeted for resume but received nil "+
				"or type-mismatched ResumeResponse; the caller must supply a *permission.ResumeResponse "+
				"via ResumeWithParams", tCtx.Name, tCtx.CallID)
	}

	decision, err := m.checker(ctx, tCtx, &schema.ToolArgument{Text: argumentsInJSON})
	if err != nil {
		return nil, fmt.Errorf(
			"permission: checker error for tool %q (call_id=%s, args=%s): %w",
			tCtx.Name, tCtx.CallID, argumentsInJSON, err)
	}
	if decision == nil {
		return nil, fmt.Errorf(
			"permission: checker returned nil ToolCallDecision for tool %q (call_id=%s); "+
				"return a valid *ToolCallDecision with Decision set to Allow, Deny, or Ask",
			tCtx.Name, tCtx.CallID)
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
			fmt.Sprintf("unknown permission decision %q; expected allow, deny, or ask", decision.Decision))}, nil
	}
}

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
			return schema.StreamReaderFromArray([]string{result.denyResult}), nil
		}
		return endpoint(ctx, result.updatedInput, opts...)
	}, nil
}

func formatDenyResult(toolName, message string) string {
	return fmt.Sprintf("Permission denied for tool %s: %s", toolName, message)
}
