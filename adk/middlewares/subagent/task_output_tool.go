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

	"github.com/cloudwego/eino/adk/internal"
	"github.com/cloudwego/eino/adk/taskstate"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

type taskOutputTool struct {
	mgr taskstate.Manager
}

type taskOutputInput struct {
	TaskID string `json:"task_id"`
}

func (t *taskOutputTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	desc := internal.SelectPrompt(internal.I18nPrompts{
		English: taskOutputToolDescription,
		Chinese: taskOutputToolDescriptionChinese,
	})
	return &schema.ToolInfo{
		Name: taskOutputToolName,
		Desc: desc,
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"task_id": {
				Type:     schema.String,
				Desc:     "The task ID to get output from",
				Required: true,
			},
		}),
	}, nil
}

func (t *taskOutputTool) InvokableRun(_ context.Context, argumentsInJSON string, _ ...tool.Option) (string, error) {
	input := &taskOutputInput{}
	if err := json.Unmarshal([]byte(argumentsInJSON), input); err != nil {
		return "", fmt.Errorf("failed to unmarshal task_output input: %w", err)
	}

	if input.TaskID == "" {
		return "", fmt.Errorf("task_id is required")
	}

	entry, ok := t.mgr.Get(input.TaskID)
	if !ok {
		return fmt.Sprintf("Task %q not found", input.TaskID), nil
	}

	return formatTaskEntry(entry), nil
}

func formatTaskEntry(entry *taskstate.Entry) string {
	result := fmt.Sprintf("Task ID: %s\nDescription: %s\nStatus: %s",
		entry.ID, entry.Description, entry.Status)

	if entry.Result != "" {
		result += fmt.Sprintf("\nResult: %s", entry.Result)
	}
	if entry.Error != "" {
		result += fmt.Sprintf("\nError: %s", entry.Error)
	}
	if entry.DoneAt != nil {
		result += fmt.Sprintf("\nCompleted at: %s", entry.DoneAt.Format("2006-01-02 15:04:05"))
	}

	return result
}
