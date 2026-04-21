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
	"errors"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

type bfaDelegateModel struct {
	generateFn func(input []*schema.Message) (*schema.Message, error)
	streamFn   func(input []*schema.Message) (*schema.StreamReader[*schema.Message], error)
}

func (m *bfaDelegateModel) Generate(_ context.Context, input []*schema.Message, _ ...model.Option) (*schema.Message, error) {
	if m.generateFn != nil {
		return m.generateFn(input)
	}
	return schema.AssistantMessage("default", nil), nil
}

func (m *bfaDelegateModel) Stream(_ context.Context, input []*schema.Message, _ ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	if m.streamFn != nil {
		return m.streamFn(input)
	}
	return schema.StreamReaderFromArray([]*schema.Message{
		schema.AssistantMessage("default", nil),
	}), nil
}

type beforeFinalAnswerHandler struct {
	BaseChatModelAgentMiddleware
	fn func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error)
}

func (h *beforeFinalAnswerHandler) BeforeFinalAnswer(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
	return h.fn(ctx, state)
}

func drainBFAIterator(iter *AsyncIterator[*AgentEvent]) []*AgentEvent {
	var events []*AgentEvent
	for {
		event, ok := iter.Next()
		if !ok {
			break
		}
		events = append(events, event)
	}
	return events
}

func drainBFAStreamEvents(events []*AgentEvent) {
	for _, event := range events {
		if event.Output != nil && event.Output.MessageOutput != nil && event.Output.MessageOutput.IsStreaming {
			sr := event.Output.MessageOutput.MessageStream
			for {
				_, err := sr.Recv()
				if err != nil {
					break
				}
			}
		}
	}
}

func TestBeforeFinalAnswer(t *testing.T) {
	t.Run("no-tools invoke: accept on first call exits immediately", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				atomic.AddInt32(&callCount, 1)
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		var hookCalls int32
		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&hookCalls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		assert.Equal(t, 1, len(events))
		assert.Nil(t, events[0].Err)
		assert.Equal(t, "answer", events[0].Output.MessageOutput.Message.Content)
		assert.Equal(t, int32(1), atomic.LoadInt32(&callCount))
		assert.Equal(t, int32(1), atomic.LoadInt32(&hookCalls))
	})

	t.Run("no-tools invoke: reject causes re-iteration with modified messages", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(input []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&callCount, 1)
				if count == 1 {
					msg := schema.AssistantMessage("partial", nil)
					msg.ResponseMeta = &schema.ResponseMeta{FinishReason: "length"}
					return msg, nil
				}
				return schema.AssistantMessage("complete answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					lastMsg := state.Messages[len(state.Messages)-1]
					if lastMsg.ResponseMeta != nil && lastMsg.ResponseMeta.FinishReason == "length" {
						state.Messages = append(state.Messages, schema.UserMessage("Please continue."))
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "complete answer", lastEvent.Output.MessageOutput.Message.Content)
		assert.Equal(t, int32(2), atomic.LoadInt32(&callCount))
	})

	t.Run("no-tools stream: reject causes re-iteration", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			streamFn: func(_ []*schema.Message) (*schema.StreamReader[*schema.Message], error) {
				count := atomic.AddInt32(&callCount, 1)
				if count == 1 {
					msg := schema.AssistantMessage("partial stream", nil)
					msg.ResponseMeta = &schema.ResponseMeta{FinishReason: "length"}
					return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
				}
				return schema.StreamReaderFromArray([]*schema.Message{
					schema.AssistantMessage("complete stream", nil),
				}), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					lastMsg := state.Messages[len(state.Messages)-1]
					if lastMsg.ResponseMeta != nil && lastMsg.ResponseMeta.FinishReason == "length" {
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{
			Messages: []Message{schema.UserMessage("hi")}, EnableStreaming: true,
		}))
		drainBFAStreamEvents(events)
		assert.Equal(t, int32(2), atomic.LoadInt32(&callCount))
	})

	t.Run("no-tools invoke: rejected answers count toward MaxIterations", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				atomic.AddInt32(&callCount, 1)
				return schema.AssistantMessage("bad answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 3,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					return ctx, RejectFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		lastEvent := events[len(events)-1]
		assert.True(t, errors.Is(lastEvent.Err, ErrExceedMaxIterations))
		assert.Equal(t, int32(3), atomic.LoadInt32(&callCount))
	})

	t.Run("no-tools invoke: hook error propagates immediately", func(t *testing.T) {
		hookErr := errors.New("hook failed")
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					return ctx, RejectFinalAnswer, state, hookErr
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		lastEvent := events[len(events)-1]
		assert.True(t, errors.Is(lastEvent.Err, hookErr))
	})

	t.Run("with-tools invoke: hook only runs on final answers, not tool calls", func(t *testing.T) {
		var generateCount int32
		var hookCalls int32

		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&generateCount, 1)
				if count == 1 {
					return schema.AssistantMessage("", []schema.ToolCall{{
						ID:       "call-1",
						Function: schema.FunctionCall{Name: "test_tool", Arguments: `{"name":"test"}`},
					}}), nil
				}
				return schema.AssistantMessage("final answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 10,
			ToolsConfig: ToolsConfig{
				ToolsNodeConfig: compose.ToolsNodeConfig{Tools: []tool.BaseTool{&fakeToolForTest{tarCount: 0}}},
			},
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&hookCalls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		drainBFAStreamEvents(events)

		var foundFinalAnswer bool
		for _, event := range events {
			if event.Err == nil && event.Output != nil && event.Output.MessageOutput != nil {
				if event.Output.MessageOutput.Message != nil && event.Output.MessageOutput.Message.Content == "final answer" {
					foundFinalAnswer = true
				}
			}
		}
		assert.True(t, foundFinalAnswer)
		assert.Equal(t, int32(1), atomic.LoadInt32(&hookCalls))
	})

	t.Run("with-tools stream: reject loops back through ChatModel", func(t *testing.T) {
		var streamCount int32

		m := &bfaDelegateModel{
			streamFn: func(_ []*schema.Message) (*schema.StreamReader[*schema.Message], error) {
				count := atomic.AddInt32(&streamCount, 1)
				if count == 1 {
					msg := schema.AssistantMessage("bad answer", nil)
					msg.ResponseMeta = &schema.ResponseMeta{FinishReason: "length"}
					return schema.StreamReaderFromArray([]*schema.Message{msg}), nil
				}
				return schema.StreamReaderFromArray([]*schema.Message{
					schema.AssistantMessage("good answer", nil),
				}), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			ToolsConfig: ToolsConfig{
				ToolsNodeConfig: compose.ToolsNodeConfig{Tools: []tool.BaseTool{&fakeToolForTest{tarCount: 0}}},
			},
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					lastMsg := state.Messages[len(state.Messages)-1]
					if lastMsg.ResponseMeta != nil && lastMsg.ResponseMeta.FinishReason == "length" {
						state.Messages = append(state.Messages, schema.UserMessage("Continue please."))
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{
			Messages: []Message{schema.UserMessage("hi")}, EnableStreaming: true,
		}))
		drainBFAStreamEvents(events)
		assert.Equal(t, int32(2), atomic.LoadInt32(&streamCount))
	})

	t.Run("no handlers means default accept behavior", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				atomic.AddInt32(&callCount, 1)
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		assert.Equal(t, 1, len(events))
		assert.Nil(t, events[0].Err)
		assert.Equal(t, int32(1), atomic.LoadInt32(&callCount))
	})

	t.Run("no-tools invoke: reject on empty content, accept on non-empty", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&callCount, 1)
				if count == 1 {
					return schema.AssistantMessage("", nil), nil
				}
				return schema.AssistantMessage("real content", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					lastMsg := state.Messages[len(state.Messages)-1]
					if lastMsg.Content == "" && len(lastMsg.ToolCalls) == 0 {
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		assert.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "real content", lastEvent.Output.MessageOutput.Message.Content)
		assert.Equal(t, int32(2), atomic.LoadInt32(&callCount))
	})

	t.Run("nil state return with accept preserves previous state", func(t *testing.T) {
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					return ctx, AcceptFinalAnswer, nil, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "answer", lastEvent.Output.MessageOutput.Message.Content)
	})

	t.Run("nil state return with reject loops without panic", func(t *testing.T) {
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 3,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					return ctx, RejectFinalAnswer, nil, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		drainBFAStreamEvents(events)
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.ErrorIs(t, lastEvent.Err, ErrExceedMaxIterations)
	})

	t.Run("multiple handlers: first reject short-circuits, skips remaining handlers", func(t *testing.T) {
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&callCount, 1)
				if count == 1 {
					return schema.AssistantMessage("bad", nil), nil
				}
				return schema.AssistantMessage("good", nil), nil
			},
		}

		var h1Calls, h2Calls int32
		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h1Calls, 1)
					if state.Messages[len(state.Messages)-1].Content == "bad" {
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h2Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "good", lastEvent.Output.MessageOutput.Message.Content)

		// h1 runs both times (reject on "bad", accept on "good")
		assert.Equal(t, int32(2), atomic.LoadInt32(&h1Calls))
		// h2 is skipped on the first call (h1 rejected), but runs on the second (h1 accepted)
		assert.Equal(t, int32(1), atomic.LoadInt32(&h2Calls))
		assert.Equal(t, int32(2), atomic.LoadInt32(&callCount))
	})

	t.Run("multiple handlers: all accept, all execute", func(t *testing.T) {
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("answer", nil), nil
			},
		}

		var h1Calls, h2Calls, h3Calls int32
		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h1Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h2Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h3Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "answer", lastEvent.Output.MessageOutput.Message.Content)
		assert.Equal(t, int32(1), atomic.LoadInt32(&h1Calls))
		assert.Equal(t, int32(1), atomic.LoadInt32(&h2Calls))
		assert.Equal(t, int32(1), atomic.LoadInt32(&h3Calls))
	})

	t.Run("multiple handlers: first handler rejects, all subsequent skipped", func(t *testing.T) {
		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("bad", nil), nil
			},
		}

		var h1Calls, h2Calls, h3Calls int32
		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 3,
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h1Calls, 1)
					return ctx, RejectFinalAnswer, state, nil
				}},
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h2Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&h3Calls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.ErrorIs(t, lastEvent.Err, ErrExceedMaxIterations)
		// h1 runs every iteration, h2 and h3 never run (always short-circuited)
		assert.Equal(t, int32(3), atomic.LoadInt32(&h1Calls))
		assert.Equal(t, int32(0), atomic.LoadInt32(&h2Calls))
		assert.Equal(t, int32(0), atomic.LoadInt32(&h3Calls))
	})

	t.Run("accept path: handler state modifications are written back", func(t *testing.T) {
		// This verifies the bug fix: accept-path state modifications were
		// previously silently dropped because state was only written back on reject.
		var callCount int32
		m := &bfaDelegateModel{
			generateFn: func(input []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&callCount, 1)
				if count == 1 {
					return schema.AssistantMessage("first answer", nil), nil
				}
				// On the second call, verify the handler's state modification persisted.
				// The handler should have appended a marker message on the first accept.
				lastUserMsg := ""
				for i := len(input) - 1; i >= 0; i-- {
					if input[i].Role == schema.User {
						lastUserMsg = input[i].Content
						break
					}
				}
				return schema.AssistantMessage("saw:"+lastUserMsg, nil), nil
			},
		}

		firstCall := true
		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			Handlers: []ChatModelAgentMiddleware{
				// First handler: on first call, modifies state AND accepts.
				// Then rejects on second call to trigger another iteration to verify state.
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					if firstCall {
						firstCall = false
						// Modify state (e.g., strip thinking tokens, add metadata)
						state.Messages = append(state.Messages, schema.UserMessage("marker-from-accept-handler"))
						return ctx, RejectFinalAnswer, state, nil
					}
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		require.NotEmpty(t, events)
		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		// The model's second call should see the marker message appended by the handler
		assert.Equal(t, "saw:marker-from-accept-handler", lastEvent.Output.MessageOutput.Message.Content)
	})

	t.Run("ReturnDirectly: accepted tool result exits normally through BeforeFinalAnswer", func(t *testing.T) {
		var hookCalls int32

		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				return schema.AssistantMessage("", []schema.ToolCall{{
					ID:       "call-1",
					Function: schema.FunctionCall{Name: "rd_tool", Arguments: `{}`},
				}}), nil
			},
		}

		rdTool := &bfaReturnDirectlyTool{name: "rd_tool", result: "tool output"}

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			ToolsConfig: ToolsConfig{
				ToolsNodeConfig: compose.ToolsNodeConfig{Tools: []tool.BaseTool{rdTool}},
				ReturnDirectly:  map[string]bool{"rd_tool": true},
			},
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&hookCalls, 1)
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		drainBFAStreamEvents(events)
		// Hook must have been called (via the ReturnDirectly path)
		assert.Equal(t, int32(1), atomic.LoadInt32(&hookCalls))
	})

	t.Run("ReturnDirectly: rejected tool result loops back to ChatModel", func(t *testing.T) {
		var generateCount int32
		var hookCalls int32

		m := &bfaDelegateModel{
			generateFn: func(_ []*schema.Message) (*schema.Message, error) {
				count := atomic.AddInt32(&generateCount, 1)
				if count == 1 {
					// First call: invoke the ReturnDirectly tool
					return schema.AssistantMessage("", []schema.ToolCall{{
						ID:       "call-1",
						Function: schema.FunctionCall{Name: "rd_tool", Arguments: `{}`},
					}}), nil
				}
				// Second call: after rejection, model produces a normal final answer
				return schema.AssistantMessage("final answer after rejection", nil), nil
			},
		}

		rdTool := &bfaReturnDirectlyTool{name: "rd_tool", result: "insufficient result"}
		firstCall := true

		agent, err := NewChatModelAgent(context.Background(), &ChatModelAgentConfig{
			Name: "test", Description: "d", Instruction: "i", Model: m,
			MaxIterations: 5,
			ToolsConfig: ToolsConfig{
				ToolsNodeConfig: compose.ToolsNodeConfig{Tools: []tool.BaseTool{rdTool}},
				ReturnDirectly:  map[string]bool{"rd_tool": true},
			},
			Handlers: []ChatModelAgentMiddleware{
				&beforeFinalAnswerHandler{fn: func(ctx context.Context, state *ChatModelAgentState) (context.Context, FinalAnswerDecision, *ChatModelAgentState, error) {
					atomic.AddInt32(&hookCalls, 1)
					if firstCall {
						firstCall = false
						// Reject the ReturnDirectly result and append feedback
						state.Messages = append(state.Messages, schema.UserMessage("Tool result insufficient, try a different approach."))
						return ctx, RejectFinalAnswer, state, nil
					}
					// Accept the model's second answer
					return ctx, AcceptFinalAnswer, state, nil
				}},
			},
		})
		require.NoError(t, err)

		events := drainBFAIterator(agent.Run(context.Background(), &AgentInput{Messages: []Message{schema.UserMessage("hi")}}))
		drainBFAStreamEvents(events)
		require.NotEmpty(t, events)

		lastEvent := events[len(events)-1]
		assert.Nil(t, lastEvent.Err)
		assert.Equal(t, "final answer after rejection", lastEvent.Output.MessageOutput.Message.Content)
		// Hook called twice: once for ReturnDirectly (reject), once for model answer (accept)
		assert.Equal(t, int32(2), atomic.LoadInt32(&hookCalls))
		// Model called twice: first produced tool call, second produced final answer
		assert.Equal(t, int32(2), atomic.LoadInt32(&generateCount))
	})
}

// bfaReturnDirectlyTool is a simple invokable tool for BeforeFinalAnswer tests.
type bfaReturnDirectlyTool struct {
	name   string
	result string
}

func (t *bfaReturnDirectlyTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: t.name,
		Desc: "A tool that returns directly for BFA testing",
	}, nil
}

func (t *bfaReturnDirectlyTool) InvokableRun(_ context.Context, _ string, _ ...tool.Option) (string, error) {
	return t.result, nil
}
