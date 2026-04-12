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

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/internal"
	"github.com/cloudwego/eino/adk/taskstate"
	"github.com/cloudwego/eino/components/tool"
)

// Config configures the subagent middleware.
type Config struct {
	// SubAgents is the list of agents available for spawning.
	// Each agent must have a unique name. Required.
	SubAgents []adk.Agent

	// AgentToolName overrides the name of the agent-spawning tool.
	// When empty, defaults to "agent".
	AgentToolName string

	// TaskStateMgr is an optional task state manager for background task support.
	// When nil, only foreground (blocking) agent execution is available.
	// When set, the middleware injects TaskOutput and TaskStop tools in addition to
	// the Agent tool, and the Agent tool gains a run_in_background parameter.
	TaskStateMgr taskstate.Manager

	// TaskToolDescriptionGenerator overrides the default agent tool description generator.
	// The generator receives the list of sub-agents and should return a complete tool
	// description string. When nil, defaultAgentToolDescription is used.
	TaskToolDescriptionGenerator func(ctx context.Context, subAgents []adk.Agent) (string, error)

	// CustomSystemPrompt overrides the default system prompt injected by BeforeAgent.
	// When nil, the built-in prompt (with i18n support) is used.
	CustomSystemPrompt *string
}

// Validate checks the Config for correctness.
func (c *Config) Validate() error {
	if len(c.SubAgents) == 0 {
		return fmt.Errorf("subagent: SubAgents must not be empty")
	}

	names := make(map[string]struct{}, len(c.SubAgents))
	for _, a := range c.SubAgents {
		name := a.Name(context.Background())
		if _, exists := names[name]; exists {
			return fmt.Errorf("subagent: duplicate agent name %q", name)
		}
		names[name] = struct{}{}
	}

	return nil
}

// New creates a ChatModelAgentMiddleware that injects sub-agent tools into the agent context.
//
// The middleware injects an Agent tool for spawning sub-agents. When Config.TaskStateMgr is
// provided, it also injects TaskOutput and TaskStop tools for background task management.
//
// The middleware uses context-based recursion prevention: when the parent agent's BeforeAgent
// sets a context marker, sub-agents created through this middleware will not have the sub-agent
// tools re-injected, preventing infinite nesting.
func New(ctx context.Context, config *Config) (adk.ChatModelAgentMiddleware, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	// Build sub-agent map: name → InvokableTool.
	subAgentMap := make(map[string]tool.InvokableTool, len(config.SubAgents))
	for _, a := range config.SubAgents {
		name := a.Name(ctx)
		bt := adk.NewAgentTool(ctx, a)
		it, ok := bt.(tool.InvokableTool)
		if !ok {
			return nil, fmt.Errorf("subagent: agent %q does not implement InvokableTool", name)
		}
		subAgentMap[name] = it
	}

	hasBackground := config.TaskStateMgr != nil

	toolName := config.AgentToolName
	if toolName == "" {
		toolName = agentToolName
	}

	descGen := defaultAgentToolDescription
	if config.TaskToolDescriptionGenerator != nil {
		descGen = config.TaskToolDescriptionGenerator
	}

	at := &agentTool{
		name:          toolName,
		subAgents:     subAgentMap,
		subAgentSlice: config.SubAgents,
		descGen:       descGen,
		mgr:           config.TaskStateMgr,
		hasBackground: hasBackground,
	}

	var tools []tool.BaseTool
	tools = append(tools, at)

	if hasBackground {
		tools = append(tools, &taskOutputTool{mgr: config.TaskStateMgr})
		tools = append(tools, &taskStopTool{mgr: config.TaskStateMgr})
	}

	// Build system prompt.
	var instruction string
	if config.CustomSystemPrompt != nil {
		instruction = *config.CustomSystemPrompt
	} else {
		instruction = internal.SelectPrompt(internal.I18nPrompts{
			English: agentToolPrompt,
			Chinese: agentToolPromptChinese,
		})
		if hasBackground {
			instruction += internal.SelectPrompt(internal.I18nPrompts{
				English: agentToolBackgroundPrompt,
				Chinese: agentToolBackgroundPromptChinese,
			})
		}
	}

	return &subagentMiddleware{
		tools:       tools,
		instruction: instruction,
	}, nil
}

type subagentCtxKey struct{}

type subagentMiddleware struct {
	adk.BaseChatModelAgentMiddleware
	tools       []tool.BaseTool
	instruction string
}

// BeforeAgent injects sub-agent tools and instructions into the agent context.
// Uses a context marker to prevent re-injection in nested sub-agent calls.
func (m *subagentMiddleware) BeforeAgent(ctx context.Context, runCtx *adk.ChatModelAgentContext) (context.Context, *adk.ChatModelAgentContext, error) {
	if runCtx == nil {
		return ctx, runCtx, nil
	}

	// Recursion prevention: if this context already has the marker, skip injection.
	if ctx.Value(subagentCtxKey{}) != nil {
		return ctx, runCtx, nil
	}

	nCtx := context.WithValue(ctx, subagentCtxKey{}, true)
	nRunCtx := *runCtx
	nRunCtx.Instruction += "\n" + m.instruction
	nRunCtx.Tools = append(nRunCtx.Tools, m.tools...)
	return nCtx, &nRunCtx, nil
}
