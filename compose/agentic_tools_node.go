/*
 * Copyright 2024 CloudWeGo Authors
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

package compose

import (
	"context"

	"github.com/cloudwego/eino/schema"
)

// NewAgenticToolsNode creates a new AgenticToolsNode.
// e.g.
//
//	conf := &ToolsNodeConfig{
//		Tools: []tool.BaseTool{invokableTool1, streamableTool2},
//	}
//	toolsNode, err := NewAgenticToolsNode(ctx, conf)
func NewAgenticToolsNode(ctx context.Context, conf *ToolsNodeConfig) (*AgenticToolsNode, error) {
	tn, err := NewToolNode(ctx, conf)
	if err != nil {
		return nil, err
	}
	return &AgenticToolsNode{inner: tn}, nil
}

type AgenticToolsNode struct {
	inner *ToolsNode
}

func (a *AgenticToolsNode) Invoke(ctx context.Context, input *schema.AgenticMessage, opts ...ToolsNodeOption) ([]*schema.AgenticMessage, error) {
	result, err := a.inner.Invoke(ctx, agenticMessageToToolCallMessage(input), opts...)
	if err != nil {
		return nil, err
	}
	return toolMessageToAgenticMessage(result), nil
}

func (a *AgenticToolsNode) Stream(ctx context.Context, input *schema.AgenticMessage,
	opts ...ToolsNodeOption) (*schema.StreamReader[[]*schema.AgenticMessage], error) {
	result, err := a.inner.Stream(ctx, agenticMessageToToolCallMessage(input), opts...)
	if err != nil {
		return nil, err
	}
	return streamToolMessageToAgenticMessage(result), nil
}

func agenticMessageToToolCallMessage(input *schema.AgenticMessage) *schema.Message {
	var tc []schema.ToolCall
	for _, block := range input.ContentBlocks {
		if block.Type != schema.ContentBlockTypeFunctionToolCall || block.FunctionToolCall == nil {
			continue
		}
		tc = append(tc, schema.ToolCall{
			ID: block.FunctionToolCall.CallID,
			Function: schema.FunctionCall{
				Name:      block.FunctionToolCall.Name,
				Arguments: block.FunctionToolCall.Arguments,
			},
			Extra: block.Extra,
		})
	}
	return &schema.Message{
		Role:      schema.Assistant,
		ToolCalls: tc,
	}
}

func toolMessageToAgenticMessage(input []*schema.Message) []*schema.AgenticMessage {
	var results []*schema.ContentBlock
	for _, m := range input {
		ftr := &schema.FunctionToolResult{
			CallID: m.ToolCallID,
			Name:   m.ToolName,
			Result: m.Content,
		}
		if len(m.UserInputMultiContent) > 0 {
			ftr.ContentBlocks = messageInputPartsToContentBlocks(m.UserInputMultiContent)
		}
		results = append(results, &schema.ContentBlock{
			Type:               schema.ContentBlockTypeFunctionToolResult,
			FunctionToolResult: ftr,
			Extra:              m.Extra,
		})
	}
	return []*schema.AgenticMessage{{
		Role:          schema.AgenticRoleTypeUser,
		ContentBlocks: results,
	}}
}

func streamToolMessageToAgenticMessage(input *schema.StreamReader[[]*schema.Message]) *schema.StreamReader[[]*schema.AgenticMessage] {
	return schema.StreamReaderWithConvert(input, func(t []*schema.Message) ([]*schema.AgenticMessage, error) {
		var results []*schema.ContentBlock
		for i, m := range t {
			if m == nil {
				continue
			}
			ftr := &schema.FunctionToolResult{
				CallID: m.ToolCallID,
				Name:   m.ToolName,
				Result: m.Content,
			}
			if len(m.UserInputMultiContent) > 0 {
				ftr.ContentBlocks = messageInputPartsToContentBlocks(m.UserInputMultiContent)
			}
			results = append(results, &schema.ContentBlock{
				Type:               schema.ContentBlockTypeFunctionToolResult,
				FunctionToolResult: ftr,
				StreamingMeta:      &schema.StreamingMeta{Index: i},
				Extra:              m.Extra,
			})
		}
		return []*schema.AgenticMessage{{
			Role:          schema.AgenticRoleTypeUser,
			ContentBlocks: results,
		}}, nil
	})
}

func messageInputPartsToContentBlocks(parts []schema.MessageInputPart) []*schema.ContentBlock {
	blocks := make([]*schema.ContentBlock, 0, len(parts))
	for _, p := range parts {
		var block *schema.ContentBlock
		switch p.Type {
		case schema.ChatMessagePartTypeText:
			block = schema.NewContentBlock(&schema.UserInputText{Text: p.Text})
			block.Extra = p.Extra
		case schema.ChatMessagePartTypeImageURL:
			if p.Image != nil {
				block = schema.NewContentBlock(&schema.UserInputImage{
					URL:        derefString(p.Image.URL),
					Base64Data: derefString(p.Image.Base64Data),
					MIMEType:   p.Image.MIMEType,
					Detail:     p.Image.Detail,
				})
				block.Extra = mergeExtra(p.Extra, p.Image.Extra)
			}
		case schema.ChatMessagePartTypeAudioURL:
			if p.Audio != nil {
				block = schema.NewContentBlock(&schema.UserInputAudio{
					URL:        derefString(p.Audio.URL),
					Base64Data: derefString(p.Audio.Base64Data),
					MIMEType:   p.Audio.MIMEType,
				})
				block.Extra = mergeExtra(p.Extra, p.Audio.Extra)
			}
		case schema.ChatMessagePartTypeVideoURL:
			if p.Video != nil {
				block = schema.NewContentBlock(&schema.UserInputVideo{
					URL:        derefString(p.Video.URL),
					Base64Data: derefString(p.Video.Base64Data),
					MIMEType:   p.Video.MIMEType,
				})
				block.Extra = mergeExtra(p.Extra, p.Video.Extra)
			}
		case schema.ChatMessagePartTypeFileURL:
			if p.File != nil {
				block = schema.NewContentBlock(&schema.UserInputFile{
					URL:        derefString(p.File.URL),
					Base64Data: derefString(p.File.Base64Data),
					Name:       p.File.Name,
					MIMEType:   p.File.MIMEType,
				})
				block.Extra = mergeExtra(p.Extra, p.File.Extra)
			}
		}
		if block != nil {
			blocks = append(blocks, block)
		}
	}
	return blocks
}

// mergeExtra merges two extra maps. Values in b take precedence over a.
func mergeExtra(a, b map[string]any) map[string]any {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	merged := make(map[string]any, len(a)+len(b))
	for k, v := range a {
		merged[k] = v
	}
	for k, v := range b {
		merged[k] = v
	}
	return merged
}

func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func (a *AgenticToolsNode) GetType() string { return "" }
