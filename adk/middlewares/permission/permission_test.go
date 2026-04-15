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
	"bytes"
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/internal/core"
	"github.com/cloudwego/eino/schema"
)

func TestNew(t *testing.T) {
	called := false
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		called = true
		return &ToolCallDecision{Decision: Allow}, nil
	})
	assert.NotNil(t, m)
	assert.NotNil(t, m.checker)
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		assert.Equal(t, "ReadFile", tCtx.Name)
		assert.Equal(t, `{"path":"/tmp/x"}`, argumentsInJSON)
		return &ToolCallDecision{Decision: Allow, Reason: "read-only"}, nil
	})

	tCtx := &adk.ToolContext{Name: "ReadFile", CallID: "call_1"}
	result, err := m.permissionGate(context.Background(), tCtx, `{"path":"/tmp/x"}`)
	require.NoError(t, err)
	assert.True(t, result.allowed)
	assert.Equal(t, `{"path":"/tmp/x"}`, result.updatedInput)
}

func TestPermissionGate_AllowWithUpdatedInput(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return nil, nil
	})

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_5"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	assert.Nil(t, result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil ToolCallDecision")
}

func TestPermissionGate_BeforeToolCallError(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return nil, fmt.Errorf("rule store unreachable")
	})

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_6"}
	result, err := m.permissionGate(context.Background(), tCtx, `{}`)
	assert.Nil(t, result)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "permission: checker error")
	assert.Contains(t, err.Error(), "rule store unreachable")
}

func TestWrapInvokableToolCall_Allow(t *testing.T) {
	endpointCalled := false
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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
	assert.Contains(t, err.Error(), "permission: checker error")
}

func TestMiddleware_ImplementsInterface(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
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

func TestAttack_NilBeforeToolCall(t *testing.T) {
	m := New(nil)
	require.NotNil(t, m)

	tCtx := &adk.ToolContext{Name: "SomeTool", CallID: "call_nil"}
	assert.Panics(t, func() {
		_, _ = m.permissionGate(context.Background(), tCtx, `{}`)
	})
}

func TestAttack_EmptyDenyMessage(t *testing.T) {
	result := formatDenyResult("WriteTool", "")
	assert.Equal(t, "Permission denied for tool WriteTool: ", result)

	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Deny, Message: ""}, nil
	})

	tCtx := &adk.ToolContext{Name: "WriteTool", CallID: "call_empty_deny"}
	gr, err := m.permissionGate(context.Background(), tCtx, `{}`)
	require.NoError(t, err)
	assert.False(t, gr.allowed)
	assert.Equal(t, "Permission denied for tool WriteTool: ", gr.denyResult)
}

func TestAttack_DenyWithEmptyToolName(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		assert.Equal(t, "", tCtx.Name)
		return &ToolCallDecision{Decision: Deny, Message: "no name"}, nil
	})

	tCtx := &adk.ToolContext{Name: "", CallID: "call_empty_name"}
	gr, err := m.permissionGate(context.Background(), tCtx, `{"x":1}`)
	require.NoError(t, err)
	assert.False(t, gr.allowed)
	assert.Equal(t, "Permission denied for tool : no name", gr.denyResult)
}

func TestAttack_AllowUpdatedInputEmpty(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow, UpdatedInput: ""}, nil
	})

	originalArgs := `{"important":"data"}`
	tCtx := &adk.ToolContext{Name: "MyTool", CallID: "call_empty_update"}
	gr, err := m.permissionGate(context.Background(), tCtx, originalArgs)
	require.NoError(t, err)
	assert.True(t, gr.allowed)
	assert.Equal(t, originalArgs, gr.updatedInput)

	var receivedArgs string
	endpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		receivedArgs = args
		return "ok", nil
	})

	wrapped, err := m.WrapInvokableToolCall(context.Background(), endpoint, tCtx)
	require.NoError(t, err)

	result, err := wrapped(context.Background(), originalArgs)
	require.NoError(t, err)
	assert.Equal(t, "ok", result)
	assert.Equal(t, originalArgs, receivedArgs)
}

func TestAttack_AskInfoGobSerializable(t *testing.T) {
	info := &AskInfo{
		ToolName:  "DangerTool",
		CallID:    "call_gob",
		Arguments: `{"rm":"-rf /"}`,
		Message:   "are you sure?",
	}
	state := &AskState{Info: info}

	var buf bytes.Buffer
	require.NoError(t, gob.NewEncoder(&buf).Encode(info))

	var decodedInfo AskInfo
	require.NoError(t, gob.NewDecoder(&buf).Decode(&decodedInfo))
	assert.Equal(t, info.ToolName, decodedInfo.ToolName)
	assert.Equal(t, info.CallID, decodedInfo.CallID)
	assert.Equal(t, info.Arguments, decodedInfo.Arguments)
	assert.Equal(t, info.Message, decodedInfo.Message)

	buf.Reset()
	require.NoError(t, gob.NewEncoder(&buf).Encode(state))

	var decodedState AskState
	require.NoError(t, gob.NewDecoder(&buf).Decode(&decodedState))
	require.NotNil(t, decodedState.Info)
	assert.Equal(t, info.ToolName, decodedState.Info.ToolName)
	assert.Equal(t, info.CallID, decodedState.Info.CallID)
	assert.Equal(t, info.Arguments, decodedState.Info.Arguments)
	assert.Equal(t, info.Message, decodedState.Info.Message)
}

func TestAttack_ResumeResponseEmptyUpdatedInput(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "confirm?"}, nil
	})

	originalArgs := `{"critical":"payload"}`
	tCtx := &adk.ToolContext{Name: "CriticalTool", CallID: "call_resume_empty"}

	ctx := makeCtxWithAddr()
	gr, err := m.permissionGate(ctx, tCtx, originalArgs)
	assert.Nil(t, gr)
	require.Error(t, err)
	var is *core.InterruptSignal
	require.True(t, errors.As(err, &is))

	resp := &ResumeResponse{Approved: true, UpdatedInput: ""}
	assert.True(t, resp.Approved)
	assert.Equal(t, "", resp.UpdatedInput)
}

func TestAttack_ConcurrentBeforeToolCall(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Allow}, nil
	})

	var receivedMu sync.Mutex
	received := make(map[string]string)

	endpoint := adk.InvokableToolCallEndpoint(func(ctx context.Context, args string, opts ...tool.Option) (string, error) {
		receivedMu.Lock()
		received[args] = "done"
		receivedMu.Unlock()
		return "result:" + args, nil
	})

	tCtx := &adk.ToolContext{Name: "ConcurrentTool", CallID: "call_concurrent"}
	wrapped, err := m.WrapInvokableToolCall(context.Background(), endpoint, tCtx)
	require.NoError(t, err)

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	errs := make([]error, goroutines)
	results := make([]string, goroutines)

	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			args := fmt.Sprintf(`{"id":%d}`, idx)
			results[idx], errs[idx] = wrapped(context.Background(), args)
		}(i)
	}
	wg.Wait()

	for i := 0; i < goroutines; i++ {
		assert.NoError(t, errs[i], "goroutine %d returned error", i)
		expected := fmt.Sprintf(`result:{"id":%d}`, i)
		assert.Equal(t, expected, results[i], "goroutine %d result mismatch", i)
	}

	receivedMu.Lock()
	assert.Len(t, received, goroutines)
	receivedMu.Unlock()
}

func buildResumeCtx(
	t *testing.T,
	signal *core.InterruptSignal,
	resumeData map[string]any,
) context.Context {
	t.Helper()
	id2Addr, id2State := core.SignalToPersistenceMaps(signal)
	ctx := context.Background()
	ctx = core.PopulateInterruptState(ctx, id2Addr, id2State)
	ctx = core.BatchResumeWithData(ctx, resumeData)
	ctx = core.AppendAddressSegment(ctx, "agent", "test-agent", "")
	return ctx
}

func TestE2E_AskThenResumeApproved(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "approve rm?"}, nil
	})

	tCtx := &adk.ToolContext{Name: "ShellExec", CallID: "call_e2e_1"}
	originalArgs := `{"cmd":"rm -rf /"}`

	ctx := makeCtxWithAddr()
	result, err := m.permissionGate(ctx, tCtx, originalArgs)
	assert.Nil(t, result)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))
	assert.Equal(t, originalArgs, signal.InterruptState.State.(*AskState).Info.Arguments)

	resumeCtx := buildResumeCtx(t, signal, map[string]any{
		signal.ID: &ResumeResponse{Approved: true},
	})

	result, err = m.permissionGate(resumeCtx, tCtx, originalArgs)
	require.NoError(t, err)
	assert.True(t, result.allowed)
	assert.Equal(t, originalArgs, result.updatedInput)
}

func TestE2E_AskThenResumeDenied(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "dangerous"}, nil
	})

	tCtx := &adk.ToolContext{Name: "DeleteDB", CallID: "call_e2e_deny"}
	ctx := makeCtxWithAddr()

	_, err := m.permissionGate(ctx, tCtx, `{}`)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))

	resumeCtx := buildResumeCtx(t, signal, map[string]any{
		signal.ID: &ResumeResponse{Approved: false, DenyMessage: "user said no"},
	})

	result, err := m.permissionGate(resumeCtx, tCtx, `{}`)
	require.NoError(t, err)
	assert.False(t, result.allowed)
	assert.Contains(t, result.denyResult, "user said no")
}

func TestE2E_ReInterruptNonTargetTool(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "confirm"}, nil
	})

	tCtx := &adk.ToolContext{Name: "ToolA", CallID: "call_a"}
	ctx := makeCtxWithAddr()

	_, err := m.permissionGate(ctx, tCtx, `{}`)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))

	id2Addr, id2State := core.SignalToPersistenceMaps(signal)
	resumeCtx := context.Background()
	resumeCtx = core.PopulateInterruptState(resumeCtx, id2Addr, id2State)
	resumeCtx = core.BatchResumeWithData(resumeCtx, map[string]any{
		"some_other_id": &ResumeResponse{Approved: true},
	})
	resumeCtx = core.AppendAddressSegment(resumeCtx, "agent", "test-agent", "")

	result, err := m.permissionGate(resumeCtx, tCtx, `{}`)
	assert.Nil(t, result)
	require.Error(t, err)

	var reSignal *core.InterruptSignal
	require.True(t, errors.As(err, &reSignal))
	assert.Equal(t, "ToolA", reSignal.InterruptState.State.(*AskState).Info.ToolName)
}

func TestE2E_ResumeWithUpdatedInput(t *testing.T) {
	m := New(func(ctx context.Context, tCtx *adk.ToolContext, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{Decision: Ask, Message: "sanitize?"}, nil
	})

	tCtx := &adk.ToolContext{Name: "WriteFile", CallID: "call_e2e_update"}
	originalArgs := `{"path":"/etc/passwd","content":"hacked"}`

	ctx := makeCtxWithAddr()
	_, err := m.permissionGate(ctx, tCtx, originalArgs)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))

	sanitizedArgs := `{"path":"/tmp/safe.txt","content":"ok"}`
	resumeCtx := buildResumeCtx(t, signal, map[string]any{
		signal.ID: &ResumeResponse{Approved: true, UpdatedInput: sanitizedArgs},
	})

	result, err := m.permissionGate(resumeCtx, tCtx, originalArgs)
	require.NoError(t, err)
	assert.True(t, result.allowed)
	assert.Equal(t, sanitizedArgs, result.updatedInput)
}
