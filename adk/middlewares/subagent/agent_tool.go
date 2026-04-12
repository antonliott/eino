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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/slongfield/pyfmt"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/internal"
	"github.com/cloudwego/eino/adk/taskstate"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

const (
	agentToolName      = "agent"
	taskOutputToolName = "task_output"
	taskStopToolName   = "task_stop"
)

// agentTool implements tool.InvokableTool for spawning sub-agents.
type agentTool struct {
	name          string
	subAgents     map[string]tool.InvokableTool
	subAgentSlice []adk.Agent
	descGen       func(ctx context.Context, subAgents []adk.Agent) (string, error)
	mgr           taskstate.Manager // nil = foreground only
	hasBackground bool
}

type agentInput struct {
	SubagentType string `json:"subagent_type"`
	Description  string `json:"description"`
}

type agentInputWithBackground struct {
	SubagentType    string `json:"subagent_type"`
	Description     string `json:"description"`
	RunInBackground *bool  `json:"run_in_background,omitempty"`
}

func (t *agentTool) Info(ctx context.Context) (*schema.ToolInfo, error) {
	desc, err := t.descGen(ctx, t.subAgentSlice)
	if err != nil {
		return nil, err
	}

	params := map[string]*schema.ParameterInfo{
		"subagent_type": {
			Type: schema.String,
			Desc: "The type of specialized agent to use for this task",
		},
		"description": {
			Type: schema.String,
			Desc: "A short description of the task for the agent to perform",
		},
	}

	if t.hasBackground {
		params["run_in_background"] = &schema.ParameterInfo{
			Type: schema.Boolean,
			Desc: "Set to true to run this agent in the background. You will be notified when it completes.",
		}
	}

	return &schema.ToolInfo{
		Name:        t.name,
		Desc:        desc,
		ParamsOneOf: schema.NewParamsOneOfByParams(params),
	}, nil
}

func (t *agentTool) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	var subagentType, description string
	var runInBackground bool

	if t.hasBackground {
		input := &agentInputWithBackground{}
		if err := json.Unmarshal([]byte(argumentsInJSON), input); err != nil {
			return "", fmt.Errorf("failed to unmarshal agent tool input: %w", err)
		}
		subagentType = input.SubagentType
		description = input.Description
		if input.RunInBackground != nil {
			runInBackground = *input.RunInBackground
		}
	} else {
		input := &agentInput{}
		if err := json.Unmarshal([]byte(argumentsInJSON), input); err != nil {
			return "", fmt.Errorf("failed to unmarshal agent tool input: %w", err)
		}
		subagentType = input.SubagentType
		description = input.Description
	}

	a, ok := t.subAgents[subagentType]
	if !ok {
		return "", fmt.Errorf("subagent type %q not found", subagentType)
	}

	params, err := sonic.MarshalString(map[string]string{
		"request": description,
	})
	if err != nil {
		return "", err
	}

	if runInBackground && t.mgr != nil {
		return t.runBackground(ctx, a, params, description, opts)
	}

	return a.InvokableRun(ctx, params, opts...)
}

func (t *agentTool) runBackground(ctx context.Context, a tool.InvokableTool, params, description string, opts []tool.Option) (string, error) {
	handle, err := t.mgr.Register(ctx, &taskstate.RegisterInfo{
		Description: description,
	})
	if err != nil {
		return "", fmt.Errorf("failed to register background task: %w", err)
	}

	go func() {
		result, runErr := a.InvokableRun(handle.Ctx, params, opts...)
		if runErr != nil {
			handle.Fail(runErr)
		} else {
			handle.Complete(result)
		}
	}()

	return fmt.Sprintf("Task %s launched in background: %s", handle.ID, description), nil
}

// defaultAgentToolDescription generates the agent tool description with sub-agent list.
func defaultAgentToolDescription(ctx context.Context, subAgents []adk.Agent) (string, error) {
	subAgentsDescBuilder := strings.Builder{}
	for _, a := range subAgents {
		name := a.Name(ctx)
		desc := a.Description(ctx)
		subAgentsDescBuilder.WriteString(fmt.Sprintf("- %s: %s\n", name, desc))
	}
	toolDesc := internal.SelectPrompt(internal.I18nPrompts{
		English: agentToolDescription,
		Chinese: agentToolDescriptionChinese,
	})
	return pyfmt.Fmt(toolDesc, map[string]any{
		"other_agents": subAgentsDescBuilder.String(),
	})
}
