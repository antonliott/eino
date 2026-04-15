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
		switch p.Type {
		case schema.ChatMessagePartTypeText:
			blocks = append(blocks, schema.NewContentBlock(&schema.UserInputText{Text: p.Text}))
		case schema.ChatMessagePartTypeImageURL:
			if p.Image != nil {
				blocks = append(blocks, schema.NewContentBlock(&schema.UserInputImage{
					URL:        derefString(p.Image.URL),
					Base64Data: derefString(p.Image.Base64Data),
					MIMEType:   p.Image.MIMEType,
					Detail:     p.Image.Detail,
				}))
			}
		case schema.ChatMessagePartTypeAudioURL:
			if p.Audio != nil {
				blocks = append(blocks, schema.NewContentBlock(&schema.UserInputAudio{
					URL:        derefString(p.Audio.URL),
					Base64Data: derefString(p.Audio.Base64Data),
					MIMEType:   p.Audio.MIMEType,
				}))
			}
		case schema.ChatMessagePartTypeVideoURL:
			if p.Video != nil {
				blocks = append(blocks, schema.NewContentBlock(&schema.UserInputVideo{
					URL:        derefString(p.Video.URL),
					Base64Data: derefString(p.Video.Base64Data),
					MIMEType:   p.Video.MIMEType,
				}))
			}
		case schema.ChatMessagePartTypeFileURL:
			if p.File != nil {
				blocks = append(blocks, schema.NewContentBlock(&schema.UserInputFile{
					URL:        derefString(p.File.URL),
					Base64Data: derefString(p.File.Base64Data),
					MIMEType:   p.File.MIMEType,
				}))
			}
		}
	}
	return blocks
}

func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func (a *AgenticToolsNode) GetType() string { return "" }
