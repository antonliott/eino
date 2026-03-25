package automemory

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

type fixedModel struct {
	out string
}

func (m *fixedModel) Generate(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	return schema.AssistantMessage(m.out, nil), nil
}

func (m *fixedModel) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	msg, _ := m.Generate(ctx, input)
	return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
}

func (m *fixedModel) WithTools(_ []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	return m, nil
}

func TestMiddleware_IndexInjection_Empty(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		// Model nil => topic selection disabled.
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("hi")}},
	}

	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Contains(t, out.Instruction, "# auto memory")
	require.Contains(t, out.Instruction, "## MEMORY.md")
	require.Contains(t, out.Instruction, "currently empty")
}

func TestMiddleware_TopicSelection_InsertsMemoryMessage(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()

	b.put("/mem/MEMORY.md", "- [debugging.md](debugging.md) - notes\n", now)
	b.put("/mem/debugging.md", "---\nname: Debugging\ndescription: build and test commands\ntype: project\n---\n\n# Debugging\npnpm test\n", now)
	b.put("/mem/other.md", "---\nname: Other\ndescription: unrelated\ntype: misc\n---\n", now.Add(-time.Hour))

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Model:           &fixedModel{out: `{"selected_memories":["debugging.md"]}`},
	})
	require.NoError(t, err)

	in := &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("How to run tests?")}}
	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  in,
	}

	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.NotNil(t, out.AgentInput)
	require.Len(t, out.AgentInput.Messages, 2)
	require.Equal(t, schema.User, out.AgentInput.Messages[0].Role)
	require.Contains(t, out.AgentInput.Messages[0].Content, "How to run tests?")
	require.Contains(t, out.AgentInput.Messages[1].Content, "<!-- automemory -->")
	require.NotNil(t, out.AgentInput.Messages[1].Extra)
	require.NotNil(t, out.AgentInput.Messages[1].Extra["__eino_automemory__"])
	require.Contains(t, out.AgentInput.Messages[1].Content, "Contents of /mem/debugging.md")
}

func TestMiddleware_TopicSelection_AsyncInjectsInBeforeModel(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()

	b.put("/mem/MEMORY.md", "- [debugging.md](debugging.md) - notes\n", now)
	b.put("/mem/debugging.md", "---\nname: Debugging\ndescription: build and test commands\ntype: project\n---\n\n# Debugging\npnpm test\n", now)

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Model:           &fixedModel{out: `{"selected_memories":["debugging.md"]}`},
		Read:            &ReadConfig{Mode: ReadModeAsync},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("How to run tests?")}},
	}
	ctx2, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Len(t, out.AgentInput.Messages, 1) // async doesn't inject here

	st := &adk.ChatModelAgentState{Messages: []adk.Message{schema.UserMessage("How to run tests?")}}

	require.Eventually(t, func() bool {
		_, next, err := mw.BeforeModelRewriteState(ctx2, st, nil)
		require.NoError(t, err)
		st = next
		last := st.Messages[len(st.Messages)-1]
		return len(st.Messages) == 2 && last.Extra != nil && last.Extra["__eino_automemory__"] != nil
	}, 2*time.Second, 10*time.Millisecond)
}

type panicModel struct{}

func (m *panicModel) Generate(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	panic("should not call model")
}

func (m *panicModel) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	panic("should not call model")
}

func (m *panicModel) WithTools(_ []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	return m, nil
}

type toolCallSelectionModel struct {
	calls atomic.Int32
}

func (m *toolCallSelectionModel) Generate(_ context.Context, _ []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	m.calls.Add(1)
	return schema.AssistantMessage("", []schema.ToolCall{
		{
			ID:   "select-1",
			Type: "function",
			Function: schema.FunctionCall{
				Name:      topicSelectionToolName,
				Arguments: `{"selected_memories":["debugging.md","hallucinated.md"]}`,
			},
		},
	}), nil
}

func (m *toolCallSelectionModel) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	msg, err := m.Generate(ctx, input)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
}

func (m *toolCallSelectionModel) WithTools(_ []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	return m, nil
}

type extractionModel struct {
	mu               sync.Mutex
	promptSeen       []string
	blockFirstRun    chan struct{}
	firstRunStarted  chan struct{}
	blockedOnce      atomic.Bool
	generateCallings atomic.Int32
}

type countingBackend struct {
	*InMemoryBackend
	writeCalls atomic.Int32
	mu         sync.Mutex
	paths      []string
}

func (b *countingBackend) Write(ctx context.Context, req *WriteRequest) error {
	b.writeCalls.Add(1)
	b.mu.Lock()
	b.paths = append(b.paths, req.FilePath)
	b.mu.Unlock()
	return b.InMemoryBackend.Write(ctx, req)
}

func (m *extractionModel) Generate(_ context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	m.generateCallings.Add(1)
	promptIdx := findExtractionPromptIndex(input)
	if promptIdx < 0 {
		return nil, fmt.Errorf("missing extraction prompt")
	}

	m.mu.Lock()
	m.promptSeen = append(m.promptSeen, input[promptIdx].Content)
	m.mu.Unlock()

	if hasToolMessageAfter(input, promptIdx) {
		return schema.AssistantMessage("done", nil), nil
	}

	if m.blockFirstRun != nil && !m.blockedOnce.Swap(true) {
		if m.firstRunStarted != nil {
			close(m.firstRunStarted)
		}
		<-m.blockFirstRun
	}

	payload := lastBusinessUserBeforePrompt(input, promptIdx)
	return schema.AssistantMessage("", []schema.ToolCall{
		{
			ID:   "write-topic",
			Type: "function",
			Function: schema.FunctionCall{
				Name:      "write_file",
				Arguments: fmt.Sprintf(`{"file_path":"topic.md","content":%q}`, payload),
			},
		},
		{
			ID:   "write-index",
			Type: "function",
			Function: schema.FunctionCall{
				Name:      "write_file",
				Arguments: `{"file_path":"MEMORY.md","content":"- [topic.md](topic.md)\n"}`,
			},
		},
	}), nil
}

func (m *extractionModel) Stream(ctx context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	msg, err := m.Generate(ctx, input)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
}

func (m *extractionModel) WithTools(_ []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	return m, nil
}

func findExtractionPromptIndex(input []*schema.Message) int {
	for i := len(input) - 1; i >= 0; i-- {
		if input[i] != nil && input[i].Role == schema.User && strings.Contains(input[i].Content, "memory extraction subagent") {
			return i
		}
	}
	return -1
}

func hasToolMessageAfter(input []*schema.Message, idx int) bool {
	for i := idx + 1; i < len(input); i++ {
		if input[i] != nil && input[i].Role == schema.Tool {
			switch input[i].ToolName {
			case "read_file", "glob", "write_file", "edit_file":
				return true
			default:
			}
		}
	}
	return false
}

func lastBusinessUserBeforePrompt(input []*schema.Message, promptIdx int) string {
	for i := promptIdx - 1; i >= 0; i-- {
		if input[i] == nil || input[i].Role != schema.User {
			continue
		}
		if strings.Contains(input[i].Content, "<!-- automemory -->") {
			continue
		}
		return input[i].Content
	}
	return "unknown"
}

func TestMiddleware_TopicSelection_SmallCandidateSetBypassesModel(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()

	b.put("/mem/MEMORY.md", "- [debugging.md](debugging.md)\n- [patterns.md](patterns.md)\n", now)
	b.put("/mem/debugging.md", "---\ndescription: debug notes\n---\nbody\n", now)
	b.put("/mem/patterns.md", "---\ndescription: patterns\n---\nbody\n", now)

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Model:           &panicModel{},
		Read: &ReadConfig{
			Mode: ReadModeSync,
			TopicSelection: &TopicSelectionConfig{
				TopK: 5,
			},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("How to run tests?")}},
	}

	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Len(t, out.AgentInput.Messages, 2)
	require.Contains(t, out.AgentInput.Messages[1].Content, "debugging.md")
	require.Contains(t, out.AgentInput.Messages[1].Content, "patterns.md")
}

func TestMiddleware_AfterAgent_SyncExtractionWritesMemoryFiles(t *testing.T) {
	ctx := context.Background()
	b := &countingBackend{InMemoryBackend: NewInMemoryBackend()}
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	extModel := &extractionModel{}
	var onErrStages []string
	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:  WriteModeSync,
			Model: extModel,
		},
		OnError: func(ctx context.Context, stage string, err error) {
			onErrStages = append(onErrStages, stage)
		},
	})
	require.NoError(t, err)

	state := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember alpha"),
			schema.AssistantMessage("ack", nil),
		},
	}

	_, outState, err := mw.AfterAgent(ctx, state, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)
	require.Empty(t, onErrStages)
	require.NotNil(t, outState)
	require.Equal(t, len(state.Messages), getWriteCursorFromMessages(outState.Messages))
	require.GreaterOrEqual(t, extModel.generateCallings.Load(), int32(1))
	require.GreaterOrEqual(t, b.writeCalls.Load(), int32(1))
	b.mu.Lock()
	paths := append([]string(nil), b.paths...)
	b.mu.Unlock()
	require.NotEmpty(t, paths)
	require.Contains(t, paths, "/mem/topic.md")
	require.Contains(t, paths, "/mem/MEMORY.md")

	mem, err := b.Read(ctx, &ReadRequest{FilePath: "/mem/MEMORY.md"})
	require.NoError(t, err)
	require.Contains(t, mem.Content, "topic.md")

	topic, err := b.Read(ctx, &ReadRequest{FilePath: "/mem/topic.md"})
	require.NoError(t, err)
	require.Equal(t, "remember alpha", topic.Content)

	extModel.mu.Lock()
	defer extModel.mu.Unlock()
	require.NotEmpty(t, extModel.promptSeen)
	require.Contains(t, extModel.promptSeen[0], "memory extraction subagent")
	require.Contains(t, extModel.promptSeen[0], "Memory directory: /mem")
}

func TestMiddleware_AfterAgent_SyncExtraction_IteratorHandlerCanDrain(t *testing.T) {
	ctx := context.Background()
	b := &countingBackend{InMemoryBackend: NewInMemoryBackend()}
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	extModel := &extractionModel{}
	var seen atomic.Int32
	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:  WriteModeSync,
			Model: extModel,
			HandleExtractionIterator: func(ctx context.Context, iter *adk.AsyncIterator[*adk.AgentEvent]) error {
				for {
					ev, ok := iter.Next()
					if !ok {
						return nil
					}
					if ev == nil {
						continue
					}
					seen.Add(1)
					if ev.Err != nil {
						return ev.Err
					}
				}
			},
		},
	})
	require.NoError(t, err)

	state := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember handler"),
			schema.AssistantMessage("ack", nil),
		},
	}

	_, outState, err := mw.AfterAgent(ctx, state, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)
	require.NotNil(t, outState)
	require.Greater(t, seen.Load(), int32(0))

	// Still writes memory files as usual (handler only changes event draining).
	_, err = b.Read(ctx, &ReadRequest{FilePath: "/mem/topic.md"})
	require.NoError(t, err)
}

func TestMiddleware_AfterAgent_SkipsExtractionWhenMainAgentAlreadyWroteMemory(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	extModel := &extractionModel{}
	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:  WriteModeSync,
			Model: extModel,
		},
	})
	require.NoError(t, err)

	state := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember beta"),
			schema.AssistantMessage("", []schema.ToolCall{
				{
					ID:   "call-1",
					Type: "function",
					Function: schema.FunctionCall{
						Name:      "write_file",
						Arguments: `{"file_path":"/mem/topic.md","content":"written by main agent"}`,
					},
				},
			}),
			schema.ToolMessage("ok", "call-1", schema.WithToolName("write_file")),
		},
	}

	_, outState, err := mw.AfterAgent(ctx, state, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)
	require.Equal(t, len(state.Messages), getWriteCursorFromMessages(outState.Messages))
	require.EqualValues(t, 0, extModel.generateCallings.Load())

	_, err = b.Read(ctx, &ReadRequest{FilePath: "/mem/topic.md"})
	require.Error(t, err)
}

func TestMiddleware_AfterAgent_AsyncExtractionKeepsLatestPendingSnapshot(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	blockCh := make(chan struct{})
	startedCh := make(chan struct{})
	extModel := &extractionModel{
		blockFirstRun:   blockCh,
		firstRunStarted: startedCh,
	}
	coord := &CoordinationConfig{
		SessionIDFunc: func(ctx context.Context, state *adk.ChatModelAgentState) (string, error) {
			return "session-1", nil
		},
		Coordinator: NewLocalCoordinator(),
		LockTTL:     time.Minute,
	}

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:  WriteModeAsync,
			Model: extModel,
		},
		Coordination: coord,
	})
	require.NoError(t, err)

	state1 := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember one"),
			schema.AssistantMessage("ack1", nil),
		},
	}
	_, _, err = mw.AfterAgent(ctx, state1, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)

	<-startedCh

	state2 := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember one"),
			schema.AssistantMessage("ack1", nil),
			schema.UserMessage("remember two"),
			schema.AssistantMessage("ack2", nil),
		},
	}
	_, _, err = mw.AfterAgent(ctx, state2, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)

	close(blockCh)

	require.Eventually(t, func() bool {
		topic, readErr := b.Read(ctx, &ReadRequest{FilePath: "/mem/topic.md"})
		if readErr != nil {
			return false
		}
		return topic.Content == "remember two"
	}, 2*time.Second, 10*time.Millisecond)

	cursor, ok, err := coord.Coordinator.GetCursor(ctx, "session-1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, len(state2.Messages), cursor)
}

func TestMiddleware_BeforeAgent_InstructionIdempotent_NoTopicMemory(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "line1\nline2\n", now)

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		// No topic selection model.
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("hi")}},
	}

	_, out1, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Contains(t, out1.Instruction, instructionMarker)

	// Call again with the already-injected instruction; should not duplicate.
	_, out2, err := mw.BeforeAgent(ctx, &adk.ChatModelAgentContext{
		Instruction: out1.Instruction,
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("hi again")}},
	})
	require.NoError(t, err)
	require.Equal(t, 1, strings.Count(out2.Instruction, instructionMarker))
}

func TestMiddleware_BeforeAgent_SkipsWhenMessagesAlreadyContainMemory(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
	})
	require.NoError(t, err)

	memMsg := newMemoryMessage("<!-- automemory -->\n<system-reminder>preloaded</system-reminder>")
	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("hi"), memMsg}},
	}

	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Equal(t, "base", out.Instruction)
	require.Len(t, out.AgentInput.Messages, 2)
}

func TestMiddleware_BeforeAgent_DistributedCursorSyncIntoMessageExtra(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	coord := &CoordinationConfig{
		SessionIDFunc: func(ctx context.Context, state *adk.ChatModelAgentState) (string, error) {
			return "sess-cursor", nil
		},
		Coordinator: NewLocalCoordinator(),
		LockTTL:     time.Minute,
	}
	require.NoError(t, coord.Coordinator.SetCursor(ctx, "sess-cursor", 5))

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Coordination:    coord,
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput: &adk.AgentInput{Messages: []adk.Message{
			schema.UserMessage("hi"),
			schema.AssistantMessage("ack", nil),
		}},
	}

	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	last := out.AgentInput.Messages[len(out.AgentInput.Messages)-1]
	require.NotNil(t, last.Extra)
	meta, ok := last.Extra[memoryExtraKey].(*memoryExtra)
	require.True(t, ok)
	require.Equal(t, "write_cursor", meta.Type)
	require.EqualValues(t, 5, meta.Cursor)
}

func TestMiddleware_TopicSelection_ToolCallParsingAndFiltering(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "- [debugging.md](debugging.md)\n", now)
	b.put("/mem/debugging.md", "---\ndescription: debug notes\n---\nbody\n", now)
	b.put("/mem/other.md", "---\ndescription: other\n---\nbody\n", now.Add(-time.Hour))

	selModel := &toolCallSelectionModel{}
	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Model:           selModel,
		Read: &ReadConfig{
			Mode: ReadModeSync,
			TopicSelection: &TopicSelectionConfig{
				TopK: 1,
			},
		},
	})
	require.NoError(t, err)

	runCtx := &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("How to debug?")}},
	}
	_, out, err := mw.BeforeAgent(ctx, runCtx)
	require.NoError(t, err)
	require.Len(t, out.AgentInput.Messages, 2)
	mem := out.AgentInput.Messages[1]
	require.Contains(t, mem.Content, "Contents of /mem/debugging.md")
	require.NotContains(t, mem.Content, "hallucinated.md")
	require.EqualValues(t, 1, selModel.calls.Load())
}

func TestMiddleware_TopicSelection_AsyncProtectsMemoryMessageFromMutation(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "- [debugging.md](debugging.md)\n", now)
	b.put("/mem/debugging.md", "---\ndescription: debug notes\n---\nbody\n", now)

	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Model:           &fixedModel{out: `{"selected_memories":["debugging.md"]}`},
		Read:            &ReadConfig{Mode: ReadModeAsync},
	})
	require.NoError(t, err)

	ctx2, _, err := mw.BeforeAgent(ctx, &adk.ChatModelAgentContext{
		Instruction: "base",
		AgentInput:  &adk.AgentInput{Messages: []adk.Message{schema.UserMessage("hi")}},
	})
	require.NoError(t, err)

	st := &adk.ChatModelAgentState{Messages: []adk.Message{schema.UserMessage("hi")}}

	var expected string
	require.Eventually(t, func() bool {
		_, next, callErr := mw.BeforeModelRewriteState(ctx2, st, nil)
		require.NoError(t, callErr)
		st = next
		if len(st.Messages) < 2 {
			return false
		}
		expected = st.Messages[len(st.Messages)-1].Content
		return strings.Contains(expected, "<!-- automemory -->")
	}, 2*time.Second, 10*time.Millisecond)

	// Mutate the memory message content.
	st.Messages[len(st.Messages)-1].Content = "tampered"
	_, next, err := mw.BeforeModelRewriteState(ctx2, st, nil)
	require.NoError(t, err)
	require.Equal(t, expected, next.Messages[len(next.Messages)-1].Content)
	require.NotNil(t, next.Messages[len(next.Messages)-1].Extra[memoryExtraKey])
}

func TestMiddleware_AfterAgent_SyncExtraction_SkipIndexPrompt(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	extModel := &extractionModel{}
	mw, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:      WriteModeSync,
			Model:     extModel,
			SkipIndex: true,
		},
	})
	require.NoError(t, err)

	state := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember gamma"),
			schema.AssistantMessage("ack", nil),
		},
	}
	_, _, err = mw.AfterAgent(ctx, state, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)

	extModel.mu.Lock()
	defer extModel.mu.Unlock()
	require.NotEmpty(t, extModel.promptSeen)
	require.NotContains(t, extModel.promptSeen[0], "Step 2")
}

func TestMiddleware_AfterAgent_AsyncSetsPendingSnapshotWhenLockHeld(t *testing.T) {
	ctx := context.Background()
	b := NewInMemoryBackend()
	now := time.Now()
	b.put("/mem/MEMORY.md", "", now)

	extModel := &extractionModel{}
	coord := &CoordinationConfig{
		SessionIDFunc: func(ctx context.Context, state *adk.ChatModelAgentState) (string, error) {
			return "sess-pending", nil
		},
		Coordinator: NewLocalCoordinator(),
		LockTTL:     time.Minute,
	}
	// Hold the lock.
	unlock, ok, err := coord.Coordinator.AcquireLock(ctx, "sess-pending", time.Minute)
	require.NoError(t, err)
	require.True(t, ok)

	mwI, err := New(ctx, &Config{
		MemoryDirectory: "/mem",
		MemoryBackend:   b,
		Write: &WriteConfig{
			Mode:  WriteModeAsync,
			Model: extModel,
		},
		Coordination: coord,
	})
	require.NoError(t, err)
	mw := mwI.(*middleware)

	state := &adk.ChatModelAgentState{
		Messages: []adk.Message{
			schema.UserMessage("remember pending"),
			schema.AssistantMessage("ack", nil),
		},
	}
	_, _, err = mw.AfterAgent(ctx, state, &adk.ChatModelAgentRunResult{})
	require.NoError(t, err)

	pending, err := coord.Coordinator.PopPendingSnapshot(ctx, "sess-pending")
	require.NoError(t, err)
	require.NotNil(t, pending)

	// Release and drain manually to complete write synchronously in test.
	require.NoError(t, unlock(ctx))
	unlock2, ok, err := coord.Coordinator.AcquireLock(ctx, "sess-pending", time.Minute)
	require.NoError(t, err)
	require.True(t, ok)
	mw.runExtractionDrain(ctx, "sess-pending", unlock2, pending)

	topic, err := b.Read(ctx, &ReadRequest{FilePath: "/mem/topic.md"})
	require.NoError(t, err)
	require.Equal(t, "remember pending", topic.Content)
}
