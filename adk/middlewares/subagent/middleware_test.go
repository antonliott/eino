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

package subagent

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/taskstate"
	"github.com/cloudwego/eino/schema"
)

// --- Mock Agent ---

type mockAgent struct {
	name string
	desc string
	// runFunc allows custom behavior in Run.
	runFunc func(ctx context.Context, input *adk.AgentInput) string
}

func (m *mockAgent) Name(_ context.Context) string {
	return m.name
}

func (m *mockAgent) Description(_ context.Context) string {
	return m.desc
}

func (m *mockAgent) Run(ctx context.Context, input *adk.AgentInput, options ...adk.AgentRunOption) *adk.AsyncIterator[*adk.AgentEvent] {
	iter, gen := adk.NewAsyncIteratorPair[*adk.AgentEvent]()

	result := m.desc // default: return description as result
	if m.runFunc != nil {
		result = m.runFunc(ctx, input)
	}

	gen.Send(adk.EventFromMessage(schema.UserMessage(result), nil, schema.User, ""))
	gen.Close()
	return iter
}

// --- Config Validation Tests ---

func TestConfigValidation_EmptySubAgents(t *testing.T) {
	_, err := New(context.Background(), &Config{
		SubAgents: nil,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must not be empty")
}

func TestConfigValidation_DuplicateNames(t *testing.T) {
	_, err := New(context.Background(), &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "agent1", desc: "first"},
			&mockAgent{name: "agent1", desc: "second"},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

// --- Middleware BeforeAgent Tests ---

func TestBeforeAgent_InjectsToolsAndInstruction(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "researcher", desc: "researches things"},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base instruction",
	}

	newCtx, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	// Instruction should be appended.
	assert.Contains(t, newRunCtx.Instruction, "base instruction")
	assert.Contains(t, newRunCtx.Instruction, "agent")

	// Agent tool should be injected.
	assert.Len(t, newRunCtx.Tools, 1)

	// Context should have the recursion marker.
	assert.NotNil(t, newCtx.Value(subagentCtxKey{}))
}

func TestBeforeAgent_RecursionPrevention(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "helper", desc: "helps"},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
	}

	// First call: injects tools.
	newCtx, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	assert.Len(t, newRunCtx.Tools, 1)

	// Second call with returned context: skips injection.
	_, secondRunCtx, err := mw.BeforeAgent(newCtx, runCtx)
	require.NoError(t, err)
	assert.Len(t, secondRunCtx.Tools, 0) // original runCtx, no tools added
}

func TestBeforeAgent_NilRunCtx(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "helper", desc: "helps"},
		},
	})
	require.NoError(t, err)

	newCtx, newRunCtx, err := mw.BeforeAgent(ctx, nil)
	require.NoError(t, err)
	assert.Nil(t, newRunCtx)
	assert.Equal(t, ctx, newCtx)
}

func TestBeforeAgent_WithBackground_InjectsThreeTools(t *testing.T) {
	ctx := context.Background()
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "worker", desc: "does work"},
		},
		TaskStateMgr: mgr,
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
	}

	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	// Should have agent + task_output + task_stop = 3 tools.
	assert.Len(t, newRunCtx.Tools, 3)

	// Instruction should include background prompt.
	assert.Contains(t, newRunCtx.Instruction, "background")
}

func TestBeforeAgent_CustomSystemPrompt(t *testing.T) {
	ctx := context.Background()
	customPrompt := "custom prompt"
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "helper", desc: "helps"},
		},
		CustomSystemPrompt: &customPrompt,
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
	}

	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	assert.Contains(t, newRunCtx.Instruction, "custom prompt")
}

// --- Agent Tool Tests ---

func TestAgentTool_ForegroundRouting(t *testing.T) {
	ctx := context.Background()
	a1 := &mockAgent{name: "agent1", desc: "desc of agent 1"}
	a2 := &mockAgent{name: "agent2", desc: "desc of agent 2"}

	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{a1, a2},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{}
	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	// Get the agent tool.
	require.Len(t, newRunCtx.Tools, 1)
	agentT, ok := newRunCtx.Tools[0].(interface {
		InvokableRun(ctx context.Context, argumentsInJSON string, opts ...interface{ isOption() }) (string, error)
	})
	// The tool implements InvokableTool, use the concrete type.
	_ = agentT
	_ = ok

	// Use the tool directly.
	at := newRunCtx.Tools[0].(*agentTool)

	result, err := at.InvokableRun(ctx, `{"subagent_type":"agent1","description":"test task"}`)
	require.NoError(t, err)
	assert.Equal(t, "desc of agent 1", result)

	result, err = at.InvokableRun(ctx, `{"subagent_type":"agent2","description":"test task"}`)
	require.NoError(t, err)
	assert.Equal(t, "desc of agent 2", result)
}

func TestAgentTool_NotFound(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "agent1", desc: "desc"},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{}
	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	at := newRunCtx.Tools[0].(*agentTool)
	_, err = at.InvokableRun(ctx, `{"subagent_type":"nonexistent","description":"test"}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestAgentTool_Background(t *testing.T) {
	ctx := context.Background()
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	slowAgent := &mockAgent{
		name: "slow",
		desc: "slow agent",
		runFunc: func(ctx context.Context, input *adk.AgentInput) string {
			time.Sleep(50 * time.Millisecond)
			return "slow result"
		},
	}

	mw, err := New(ctx, &Config{
		SubAgents:    []adk.Agent{slowAgent},
		TaskStateMgr: mgr,
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{}
	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	at := newRunCtx.Tools[0].(*agentTool)
	result, err := at.InvokableRun(ctx, `{"subagent_type":"slow","description":"bg task","run_in_background":true}`)
	require.NoError(t, err)
	assert.Contains(t, result, "launched in background")

	// Wait for the background task to complete.
	err = mgr.WaitAllDone(context.Background())
	require.NoError(t, err)

	// Check the notification.
	select {
	case n := <-mgr.Notifications():
		assert.Equal(t, taskstate.StatusCompleted, n.Entry.Status)
		assert.Equal(t, "slow result", n.Entry.Result)
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for notification")
	}
}

func TestAgentTool_Info(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "helper", desc: "helps with tasks"},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{}
	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	info, err := newRunCtx.Tools[0].Info(ctx)
	require.NoError(t, err)
	assert.Equal(t, agentToolName, info.Name)
	assert.Contains(t, info.Desc, "helper")
	assert.Contains(t, info.Desc, "helps with tasks")
}

func TestAgentTool_CustomName(t *testing.T) {
	ctx := context.Background()
	mw, err := New(ctx, &Config{
		SubAgents: []adk.Agent{
			&mockAgent{name: "helper", desc: "helps"},
		},
		AgentToolName: "task",
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{}
	_, newRunCtx, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)

	info, err := newRunCtx.Tools[0].Info(ctx)
	require.NoError(t, err)
	assert.Equal(t, "task", info.Name)
}

// --- TaskOutput Tool Tests ---

func TestTaskOutputTool(t *testing.T) {
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	h, err := mgr.Register(context.Background(), &taskstate.RegisterInfo{Description: "test task"})
	require.NoError(t, err)
	h.Complete("task result")

	tool := &taskOutputTool{mgr: mgr}
	result, err := tool.InvokableRun(context.Background(), fmt.Sprintf(`{"task_id":"%s"}`, h.ID))
	require.NoError(t, err)
	assert.Contains(t, result, "test task")
	assert.Contains(t, result, "task result")
	assert.Contains(t, result, "completed")
}

func TestTaskOutputTool_NotFound(t *testing.T) {
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	tool := &taskOutputTool{mgr: mgr}
	result, err := tool.InvokableRun(context.Background(), `{"task_id":"nonexistent"}`)
	require.NoError(t, err)
	assert.Contains(t, result, "not found")
}

// --- TaskStop Tool Tests ---

func TestTaskStopTool(t *testing.T) {
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	h, err := mgr.Register(context.Background(), &taskstate.RegisterInfo{Description: "running task"})
	require.NoError(t, err)

	tool := &taskStopTool{mgr: mgr}
	result, err := tool.InvokableRun(context.Background(), fmt.Sprintf(`{"task_id":"%s"}`, h.ID))
	require.NoError(t, err)
	assert.Contains(t, result, "Successfully stopped")

	// Verify the task is cancelled.
	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, taskstate.StatusCancelled, entry.Status)
}

func TestTaskStopTool_AlreadyDone(t *testing.T) {
	mgr := taskstate.NewInMemoryManager()
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = mgr.Close(closeCtx)
	}()

	h, err := mgr.Register(context.Background(), &taskstate.RegisterInfo{Description: "done task"})
	require.NoError(t, err)
	h.Complete("done")

	tool := &taskStopTool{mgr: mgr}
	result, err := tool.InvokableRun(context.Background(), fmt.Sprintf(`{"task_id":"%s"}`, h.ID))
	require.NoError(t, err)
	assert.Contains(t, result, "Failed to stop")
}
