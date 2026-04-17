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

// Package permission provides a tool permission middleware for ChatModelAgent.
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
	Decision Decision

	// Message is the user-facing reason for deny/ask.
	Message string

	// UpdatedInput optionally replaces the original tool input for allow decisions.
	UpdatedInput string

	// Reason records where the decision came from for debugging or telemetry.
	Reason string
}

type BeforeToolCall func(ctx context.Context, toolName string, argumentsInJSON string) (*ToolCallDecision, error)

// AskInfo is the user-facing interrupt payload exposed to the caller.
type AskInfo struct {
	ToolName  string
	CallID    string
	Arguments string
	Message   string
}

// AskState is the persisted state used to re-interrupt untargeted resume attempts.
type AskState struct {
	Info *AskInfo
}

// ResumeResponse is the decision payload injected through Runner.ResumeWithParams.
type ResumeResponse struct {
	Approved     bool
	UpdatedInput string
	DenyMessage  string
}

type PermissionMiddleware struct {
	*adk.BaseChatModelAgentMiddleware
	beforeToolCall BeforeToolCall
}

func NewPermissionMiddleware(beforeToolCall BeforeToolCall) *PermissionMiddleware {
	return &PermissionMiddleware{
		BaseChatModelAgentMiddleware: &adk.BaseChatModelAgentMiddleware{},
		beforeToolCall:               beforeToolCall,
	}
}

type gateResult struct {
	allowed      bool
	denyResult   string
	updatedInput string
}

func (m *PermissionMiddleware) permissionGate(
	ctx context.Context,
	tCtx *adk.ToolContext,
	argumentsInJSON string,
) (*gateResult, error) {
	toolName := toolNameFromContext(tCtx)

	// Phase 1: resume handling.
	wasInterrupted, hasState, savedState := tool.GetInterruptState[*AskState](ctx)
	isTarget, hasData, response := tool.GetResumeContext[*ResumeResponse](ctx)

	if wasInterrupted && !isTarget {
		if !hasState || savedState == nil {
			return nil, fmt.Errorf("permission: tool %s was interrupted but AskState is missing", toolName)
		}
		return nil, tool.StatefulInterrupt(ctx, savedState.Info, savedState)
	}

	if isTarget && hasData {
		if response == nil {
			return nil, fmt.Errorf("permission: tool %s was targeted for resume but received nil or type-mismatched ResumeResponse", toolName)
		}
		if !response.Approved {
			return &gateResult{denyResult: formatDenyResult(toolName, response.DenyMessage)}, nil
		}
		input := argumentsInJSON
		if response.UpdatedInput != "" {
			input = response.UpdatedInput
		}
		return &gateResult{allowed: true, updatedInput: input}, nil
	}

	if isTarget && !hasData {
		return nil, fmt.Errorf("permission: tool %s was targeted for resume but received nil or type-mismatched ResumeResponse", toolName)
	}

	// Phase 2: first call -> BeforeToolCall.
	if m.beforeToolCall == nil {
		return nil, fmt.Errorf("permission: BeforeToolCall is nil for tool %s", toolName)
	}

	decision, err := m.beforeToolCall(ctx, toolName, argumentsInJSON)
	if err != nil {
		return nil, fmt.Errorf("permission check failed: %w", err)
	}
	if decision == nil {
		return nil, fmt.Errorf("permission: BeforeToolCall returned nil decision for tool %s", toolName)
	}

	switch decision.Decision {
	case Allow:
		input := argumentsInJSON
		if decision.UpdatedInput != "" {
			input = decision.UpdatedInput
		}
		return &gateResult{allowed: true, updatedInput: input}, nil

	case Deny:
		return &gateResult{denyResult: formatDenyResult(toolName, decision.Message)}, nil

	case Ask:
		info := &AskInfo{
			ToolName:  toolName,
			CallID:    callIDFromContext(tCtx),
			Arguments: argumentsInJSON,
			Message:   decision.Message,
		}
		state := &AskState{Info: info}
		return nil, tool.StatefulInterrupt(ctx, info, state)

	default:
		return &gateResult{
			denyResult: formatDenyResult(toolName, fmt.Sprintf("unknown permission decision %q", decision.Decision)),
		}, nil
	}
}

func (m *PermissionMiddleware) WrapInvokableToolCall(
	_ context.Context,
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

func (m *PermissionMiddleware) WrapStreamableToolCall(
	_ context.Context,
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

func (m *PermissionMiddleware) WrapEnhancedInvokableToolCall(
	_ context.Context,
	endpoint adk.EnhancedInvokableToolCallEndpoint,
	tCtx *adk.ToolContext,
) (adk.EnhancedInvokableToolCallEndpoint, error) {
	return func(ctx context.Context, toolArgument *schema.ToolArgument, opts ...tool.Option) (*schema.ToolResult, error) {
		result, err := m.permissionGate(ctx, tCtx, toolArgumentText(toolArgument))
		if err != nil {
			return nil, err
		}
		if !result.allowed {
			return textToolResult(result.denyResult), nil
		}
		return endpoint(ctx, updateToolArgument(toolArgument, result.updatedInput), opts...)
	}, nil
}

func (m *PermissionMiddleware) WrapEnhancedStreamableToolCall(
	_ context.Context,
	endpoint adk.EnhancedStreamableToolCallEndpoint,
	tCtx *adk.ToolContext,
) (adk.EnhancedStreamableToolCallEndpoint, error) {
	return func(ctx context.Context, toolArgument *schema.ToolArgument, opts ...tool.Option) (*schema.StreamReader[*schema.ToolResult], error) {
		result, err := m.permissionGate(ctx, tCtx, toolArgumentText(toolArgument))
		if err != nil {
			return nil, err
		}
		if !result.allowed {
			sr, sw := schema.Pipe[*schema.ToolResult](1)
			sw.Send(textToolResult(result.denyResult), nil)
			sw.Close()
			return sr, nil
		}
		return endpoint(ctx, updateToolArgument(toolArgument, result.updatedInput), opts...)
	}, nil
}

func formatDenyResult(toolName, message string) string {
	return fmt.Sprintf("Permission denied for tool %s: %s", toolName, message)
}

func textToolResult(text string) *schema.ToolResult {
	return &schema.ToolResult{
		Parts: []schema.ToolOutputPart{
			{Type: schema.ToolPartTypeText, Text: text},
		},
	}
}

func updateToolArgument(toolArgument *schema.ToolArgument, updatedInput string) *schema.ToolArgument {
	if updatedInput == "" {
		return toolArgument
	}
	if toolArgument == nil {
		return &schema.ToolArgument{Text: updatedInput}
	}
	copied := *toolArgument
	copied.Text = updatedInput
	return &copied
}

func toolArgumentText(toolArgument *schema.ToolArgument) string {
	if toolArgument == nil {
		return ""
	}
	return toolArgument.Text
}

func toolNameFromContext(tCtx *adk.ToolContext) string {
	if tCtx == nil {
		return ""
	}
	return tCtx.Name
}

func callIDFromContext(tCtx *adk.ToolContext) string {
	if tCtx == nil {
		return ""
	}
	return tCtx.CallID
}
