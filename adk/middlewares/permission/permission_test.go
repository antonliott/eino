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
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/internal/core"
	"github.com/cloudwego/eino/schema"
)

func TestWrapInvokableToolCallAllowUpdatedInput(t *testing.T) {
	var gotInput string
	mw := NewPermissionMiddleware(func(_ context.Context, toolName, argumentsInJSON string) (*ToolCallDecision, error) {
		assert.Equal(t, "write_file", toolName)
		assert.Equal(t, `{"path":"a.txt"}`, argumentsInJSON)
		return &ToolCallDecision{
			Decision:     Allow,
			UpdatedInput: `{"path":"b.txt"}`,
		}, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, argumentsInJSON string, _ ...tool.Option) (string, error) {
		gotInput = argumentsInJSON
		return "ok", nil
	}, &adk.ToolContext{Name: "write_file", CallID: "call-1"})
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{"path":"a.txt"}`)
	require.NoError(t, err)
	assert.Equal(t, "ok", result)
	assert.Equal(t, `{"path":"b.txt"}`, gotInput)
}

func TestWrapInvokableToolCallDenyReturnsToolResult(t *testing.T) {
	called := false
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Deny,
			Message:  "blocked by policy",
		}, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (string, error) {
		called = true
		return "should not run", nil
	}, &adk.ToolContext{Name: "delete_file", CallID: "call-2"})
	require.NoError(t, err)

	result, err := wrapped(context.Background(), `{}`)
	require.NoError(t, err)
	assert.Equal(t, "Permission denied for tool delete_file: blocked by policy", result)
	assert.False(t, called)
}

func TestWrapInvokableToolCallAskInterrupts(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, _, argumentsInJSON string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Ask,
			Message:  "needs approval",
			Reason:   argumentsInJSON,
		}, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (string, error) {
		t.Fatal("endpoint should not run when permission asks")
		return "", nil
	}, &adk.ToolContext{Name: "bash", CallID: "call-3"})
	require.NoError(t, err)

	_, err = wrapped(context.Background(), `{"cmd":"rm -rf /tmp/x"}`)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))

	info, ok := signal.Info.(*AskInfo)
	require.True(t, ok)
	assert.Equal(t, &AskInfo{
		ToolName:  "bash",
		CallID:    "call-3",
		Arguments: `{"cmd":"rm -rf /tmp/x"}`,
		Message:   "needs approval",
	}, info)

	state, ok := signal.State.(*AskState)
	require.True(t, ok)
	assert.Equal(t, info, state.Info)
}

func TestWrapInvokableToolCallResumeApprovedWithUpdatedInput(t *testing.T) {
	var called bool
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		t.Fatal("BeforeToolCall should not run on targeted approved resume")
		return nil, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, argumentsInJSON string, _ ...tool.Option) (string, error) {
		called = true
		return argumentsInJSON, nil
	}, &adk.ToolContext{Name: "edit_file", CallID: "call-4"})
	require.NoError(t, err)

	ctx := resumedToolContext(
		&adk.ToolContext{Name: "edit_file", CallID: "call-4"},
		&AskState{Info: &AskInfo{ToolName: "edit_file", CallID: "call-4", Arguments: `{"orig":true}`}},
		map[string]any{
			interruptID: &ResumeResponse{Approved: true, UpdatedInput: `{"patched":true}`},
		},
	)

	result, err := wrapped(ctx, `{"orig":true}`)
	require.NoError(t, err)
	assert.True(t, called)
	assert.Equal(t, `{"patched":true}`, result)
}

func TestWrapInvokableToolCallResumeDeniedReturnsToolResult(t *testing.T) {
	var called bool
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		t.Fatal("BeforeToolCall should not run on targeted denied resume")
		return nil, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (string, error) {
		called = true
		return "unexpected", nil
	}, &adk.ToolContext{Name: "deploy", CallID: "call-5"})
	require.NoError(t, err)

	ctx := resumedToolContext(
		&adk.ToolContext{Name: "deploy", CallID: "call-5"},
		&AskState{Info: &AskInfo{ToolName: "deploy", CallID: "call-5"}},
		map[string]any{
			interruptID: &ResumeResponse{Approved: false, DenyMessage: "not approved"},
		},
	)

	result, err := wrapped(ctx, `{}`)
	require.NoError(t, err)
	assert.Equal(t, "Permission denied for tool deploy: not approved", result)
	assert.False(t, called)
}

func TestWrapInvokableToolCallResumeTargetWithoutTypedResponseFailsClosed(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		t.Fatal("BeforeToolCall should not run when resume payload is invalid")
		return nil, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (string, error) {
		t.Fatal("endpoint should not run when resume payload is invalid")
		return "", nil
	}, &adk.ToolContext{Name: "secure_op", CallID: "call-6"})
	require.NoError(t, err)

	ctx := resumedToolContext(
		&adk.ToolContext{Name: "secure_op", CallID: "call-6"},
		&AskState{Info: &AskInfo{ToolName: "secure_op", CallID: "call-6"}},
		map[string]any{
			interruptID: "bad payload",
		},
	)

	_, err = wrapped(ctx, `{}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "permission: tool secure_op was targeted for resume but received nil or type-mismatched ResumeResponse")
}

func TestWrapInvokableToolCallUntargetedResumeReinterrupts(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		t.Fatal("BeforeToolCall should not run when an untargeted interrupt is re-issued")
		return nil, nil
	})

	wrapped, err := mw.WrapInvokableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (string, error) {
		t.Fatal("endpoint should not run when another sibling is resumed")
		return "", nil
	}, &adk.ToolContext{Name: "tool_b", CallID: "call-7"})
	require.NoError(t, err)

	savedInfo := &AskInfo{
		ToolName:  "tool_b",
		CallID:    "call-7",
		Arguments: `{"x":1}`,
		Message:   "still waiting",
	}
	ctx := resumedToolContext(
		&adk.ToolContext{Name: "tool_b", CallID: "call-7"},
		&AskState{Info: savedInfo},
		map[string]any{
			"some-other-interrupt": &ResumeResponse{Approved: true},
		},
	)

	_, err = wrapped(ctx, `{"x":1}`)
	require.Error(t, err)

	var signal *core.InterruptSignal
	require.True(t, errors.As(err, &signal))
	info, ok := signal.Info.(*AskInfo)
	require.True(t, ok)
	assert.Equal(t, savedInfo, info)
}

func TestWrapStreamableToolCallDenyReturnsSingleChunk(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Deny,
			Message:  "stream blocked",
		}, nil
	})

	wrapped, err := mw.WrapStreamableToolCall(context.Background(), func(_ context.Context, _ string, _ ...tool.Option) (*schema.StreamReader[string], error) {
		t.Fatal("endpoint should not run on deny")
		return nil, nil
	}, &adk.ToolContext{Name: "stream_tool", CallID: "call-8"})
	require.NoError(t, err)

	stream, err := wrapped(context.Background(), `{}`)
	require.NoError(t, err)
	defer stream.Close()

	chunk, err := stream.Recv()
	require.NoError(t, err)
	assert.Equal(t, "Permission denied for tool stream_tool: stream blocked", chunk)

	_, err = stream.Recv()
	assert.Equal(t, io.EOF, err)
}

func TestWrapEnhancedInvokableToolCallAllowUpdatedInput(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, toolName, argumentsInJSON string) (*ToolCallDecision, error) {
		assert.Equal(t, "enhanced_tool", toolName)
		assert.Equal(t, `{"orig":true}`, argumentsInJSON)
		return &ToolCallDecision{
			Decision:     Allow,
			UpdatedInput: `{"new":true}`,
		}, nil
	})

	wrapped, err := mw.WrapEnhancedInvokableToolCall(context.Background(), func(_ context.Context, toolArgument *schema.ToolArgument, _ ...tool.Option) (*schema.ToolResult, error) {
		return &schema.ToolResult{
			Parts: []schema.ToolOutputPart{{Type: schema.ToolPartTypeText, Text: toolArgument.Text}},
		}, nil
	}, &adk.ToolContext{Name: "enhanced_tool", CallID: "call-9"})
	require.NoError(t, err)

	result, err := wrapped(context.Background(), &schema.ToolArgument{Text: `{"orig":true}`})
	require.NoError(t, err)
	require.Len(t, result.Parts, 1)
	assert.Equal(t, `{"new":true}`, result.Parts[0].Text)
}

func TestWrapEnhancedStreamableToolCallDenyReturnsToolResultChunk(t *testing.T) {
	mw := NewPermissionMiddleware(func(_ context.Context, _, _ string) (*ToolCallDecision, error) {
		return &ToolCallDecision{
			Decision: Deny,
			Message:  "enhanced blocked",
		}, nil
	})

	wrapped, err := mw.WrapEnhancedStreamableToolCall(context.Background(), func(_ context.Context, _ *schema.ToolArgument, _ ...tool.Option) (*schema.StreamReader[*schema.ToolResult], error) {
		t.Fatal("endpoint should not run on deny")
		return nil, nil
	}, &adk.ToolContext{Name: "enhanced_stream_tool", CallID: "call-10"})
	require.NoError(t, err)

	stream, err := wrapped(context.Background(), &schema.ToolArgument{Text: `{}`})
	require.NoError(t, err)
	defer stream.Close()

	chunk, err := stream.Recv()
	require.NoError(t, err)
	require.Len(t, chunk.Parts, 1)
	assert.Equal(t, "Permission denied for tool enhanced_stream_tool: enhanced blocked", chunk.Parts[0].Text)
}

func resumedToolContext(tCtx *adk.ToolContext, state *AskState, resumeData map[string]any) context.Context {
	ctx := context.Background()
	addr := core.Address{
		{
			Type:  core.AddressSegmentType("tool"),
			ID:    tCtx.Name,
			SubID: tCtx.CallID,
		},
	}
	ctx = core.PopulateInterruptState(ctx,
		map[string]core.Address{interruptID: addr},
		map[string]core.InterruptState{interruptID: {State: state}},
	)
	if resumeData != nil {
		ctx = core.BatchResumeWithData(ctx, resumeData)
	}
	return core.AppendAddressSegment(ctx, core.AddressSegmentType("tool"), tCtx.Name, tCtx.CallID)
}

const interruptID = "permission-interrupt-id"
