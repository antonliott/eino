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

package permission

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/internal/core"
	"github.com/cloudwego/eino/schema"
)

func TestNewMiddleware(t *testing.T) {
	called := false
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		called = true
		return &ToolCallDecision{Decision: Allow}, nil
	})
	assert.NotNil(t, m)
	assert.NotNil(t, m.beforeToolCall)
	assert.NotNil(t, m.BaseChatModelAgentMiddleware)
	assert.False(t, called)
}

func TestFormatDenyResult(t *testing.T) {
	result := formatDenyResult("WriteFile", "destructive operation blocked")
	assert.Equal(t, "Permission denied for tool WriteFile: destructive operation blocked", result)
}

func makeCtxWithAddr() context.Context {
	ctx := context.Background()
	return core.AppendAddressSegment(ctx, "agent", "test-agent", "")
}

func TestPermissionGate_Allow(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		assert.Equal(t, "ReadFile", toolName)
		assert.Equal(t, `{"path":"/tmp/x"}`, args)
		return &ToolCallDecision{Decision: Allow, Reason: "read-only"}, nil
	})

	tCtx := &adk.ToolContext{Name: "ReadFile", CallID: "call_1"}
	result, err := m.permissionGate(context.Background(), tCtx, `{"path":"/tmp/x"}`)
	require.NoError(t, err)
	assert.True(t, result.allowed)
	assert.Equal(t, `{"path":"/tmp/x"}`, result.updatedInput)
}

func TestPermissionGate_AllowWithUpdatedInput(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision:     Allow,
			UpdatedInput: `{"path":"/tmp/safe"}`,
		}, nil
	})

	tCtx := &adk.ToolContext{Name: "ReadFile", CallID: "call_1"}
	result, err := m.permissionGate(context.Background(), tCtx, `{"path":"/tmp/danger"}`)
	require.NoError(t, err)
	assert.True(t, result.allowed)
	assert.Equal(t, `{"path":"/tmp/safe"}`, result.updatedInput)
}

func TestPermissionGate_Deny(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Deny,
			Message:  "operation not allowed",
			Reason:   "policy",
		}, nil
	})

	tCtx := &adk.ToolContext{Name: "DeleteFile", CallID: "call_2"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	require.NoError(t, err)
	assert.False(t, result.allowed)
	assert.Equal(t, "Permission denied for tool DeleteFile: operation not allowed", result.denyResult)
}

func TestPermissionGate_Ask(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Ask,
			Message:  "requires approval",
		}, nil
	})

	tCtx := &adk.ToolContext{Name: "Execute", CallID: "call_3"}
	ctx := makeCtxWithAddr()
	result, err := m.permissionGate(ctx, tCtx, `{"cmd":"rm -rf /"}`)
	assert.Nil(t, result)
	require.Error(t, err)

	var is *core.InterruptSignal
	assert.True(t, errors.As(err, &is))
}

func TestPermissionGate_UnknownDecision(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: "maybe"}, nil
	})

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_4"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	require.NoError(t, err)
	assert.False(t, result.allowed)
	assert.Contains(t, result.denyResult, "unknown permission decision")
	assert.Contains(t, result.denyResult, "maybe")
}

func TestPermissionGate_NilDecision(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return nil, nil
	})

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_5"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	assert.Nil(t, result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil decision")
}

func TestPermissionGate_BeforeToolCallError(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return nil, fmt.Errorf("rule store unreachable")
	})

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_6"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	assert.Nil(t, result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "permission check failed")
	assert.Contains(t, err.Error(), "rule store unreachable")
}

func TestWrapInvokableToolCall_Allow(t *testing.T) {
	endpointCalled := false
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow}, nil
	})

	originalEndpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		endpointCalled = true
		return "tool result: " + args, nil
	})

	tCtx := &adk.ToolContext{Name: "MyTool", CallID: "call_1"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{"key":"value"}`)
	require.NoError(t, err)
	assert.True(t, endpointCalled)
	assert.Equal(t, `tool result: {"key":"value"}`, result)
}

func TestWrapInvokableToolCall_AllowWithUpdatedInput(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow, UpdatedInput: `{"sanitized":true}`}, nil
	})

	var receivedArgs string
	originalEndpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		receivedArgs = args
		return "ok", nil
	})

	tCtx := &adk.ToolContext{Name: "MyTool", CallID: "call_1"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{"original":true}`)
	require.NoError(t, err)
	assert.Equal(t, `{"sanitized":true}`, receivedArgs)
	assert.Equal(t, "ok", result)
}

func TestWrapInvokableToolCall_Deny(t *testing.T) {
	endpointCalled := false
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Deny, Message: "blocked"}, nil
	})

	originalEndpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		endpointCalled = true
		return "should not reach", nil
	})

	tCtx := &adk.ToolContext{Name: "MyTool", CallID: "call_1"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{}`)
	require.NoError(t, err)
	assert.False(t, endpointCalled)
	assert.Equal(t, "Permission denied for tool MyTool: blocked", result)
}

func TestWrapInvokableToolCall_Ask(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "need approval"}, nil
	})

	originalEndpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		return "should not reach", nil
	})

	tCtx := &adk.ToolContext{Name: "DangerTool", CallID: "call_1"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	ctx := makeCtxWithAddr()
	result, err := wrapped(ctx, `{"danger":true}`)
	assert.Equal(t, "", result)
	require.Error(t, err)

	var is *core.InterruptSignal
	assert.True(t, errors.As(err, &is))
}

func TestWrapStreamableToolCall_Allow(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow}, nil
	})

	originalEndpoint := adk.StreamableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (*schema.StreamReader[string], error) {
		sr, sw := schema.Pipe[string](1)
		sw.Send("stream chunk: "+args, nil)
		sw.Close()
		return sr, nil
	})

	tCtx := &adk.ToolContext{Name: "StreamTool", CallID: "call_1"}
	wrapped, err := m.WrapStreamableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	sr, err := wrapped(context.Background(), `{"key":"val"}`)
	require.NoError(t, err)
	require.NotNil(t, sr)

	chunk, err := sr.Recv()
	require.NoError(t, err)
	assert.Equal(t, `stream chunk: {"key":"val"}`, chunk)

	_, err = sr.Recv()
	assert.Equal(t, io.EOF, err)
}

func TestWrapStreamableToolCall_Deny(t *testing.T) {
	endpointCalled := false
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Deny, Message: "stream blocked"}, nil
	})

	originalEndpoint := adk.StreamableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (*schema.StreamReader[string], error) {
		endpointCalled = true
		return nil, nil
	})

	tCtx := &adk.ToolContext{Name: "StreamTool", CallID: "call_1"}
	wrapped, err := m.WrapStreamableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	sr, err := wrapped(context.Background(), `{}`)
	require.NoError(t, err)
	assert.False(t, endpointCalled)
	require.NotNil(t, sr)

	chunk, err := sr.Recv()
	require.NoError(t, err)
	assert.Equal(t, "Permission denied for tool StreamTool: stream blocked", chunk)

	_, err = sr.Recv()
	assert.Equal(t, io.EOF, err)
}

func TestWrapStreamableToolCall_Ask(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "stream needs approval"}, nil
	})

	originalEndpoint := adk.StreamableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (*schema.StreamReader[string], error) {
		return nil, nil
	})

	tCtx := &adk.ToolContext{Name: "StreamTool", CallID: "call_1"}
	wrapped, err := m.WrapStreamableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	ctx := makeCtxWithAddr()
	sr, err := wrapped(ctx, `{}`)
	assert.Nil(t, sr)
	require.Error(t, err)

	var is *core.InterruptSignal
	assert.True(t, errors.As(err, &is))
}

func TestWrapInvokableToolCall_BeforeToolCallError(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return nil, fmt.Errorf("infra failure")
	})

	originalEndpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		return "should not reach", nil
	})

	tCtx := &adk.ToolContext{Name: "MyTool", CallID: "call_1"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), originalEndpoint, tCtx)
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{}`)
	assert.Equal(t, "", result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "permission check failed")
}

func TestMiddleware_ImplementsInterface(t *testing.T) {
	m := NewMiddleware(func(ctx context.Context, toolName, args string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow}, nil
	})
	var _ adk.ChatModelAgentMiddleware = m
}

func TestDecisionConstants(t *testing.T) {
	assert.Equal(t, Decision("allow"), Allow)
	assert.Equal(t, Decision("deny"), Deny)
	assert.Equal(t, Decision("ask"), Ask)
}

func TestAskInfoFields(t *testing.T) {
	info := &AskInfo{
		ToolName:  "MyTool",
		CallID:    "call_1",
		Arguments: `{"key":"value"}`,
		Message:   "requires approval",
	}
	assert.Equal(t, "MyTool", info.ToolName)
	assert.Equal(t, "call_1", info.CallID)
	assert.Equal(t, `{"key":"value"}`, info.Arguments)
	assert.Equal(t, "requires approval", info.Message)
}

func TestResumeResponse_Approved(t *testing.T) {
	resp := &ResumeResponse{
		Approved:     true,
		UpdatedInput: `{"modified":true}`,
	}
	assert.True(t, resp.Approved)
	assert.Equal(t, `{"modified":true}`, resp.UpdatedInput)
}

func TestResumeResponse_Denied(t *testing.T) {
	resp := &ResumeResponse{
		Approved:    false,
		DenyMessage: "user rejected",
	}
	assert.False(t, resp.Approved)
	assert.Equal(t, "user rejected", resp.DenyMessage)
}
