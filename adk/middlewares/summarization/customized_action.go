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

package summarization

import (
	"github.com/cloudwego/eino/adk"
)

type CustomizedAction struct {
	// Type is the action type.
	Type ActionType `json:"type"`

	// Before is set when Type is ActionTypeBeforeSummarize.
	// Emitted after trigger condition is met, before calling model to generate summary.
	Before *BeforeSummarizeAction `json:"before,omitempty"`

	// After is set when Type is ActionTypeAfterSummarize.
	// Emitted after summarization.
	After *AfterSummarizeAction `json:"after,omitempty"`

	// GenerateSummary is set when Type is ActionTypeGenerateSummary.
	// Emitted after each model generate attempt during summarization.
	GenerateSummary *GenerateSummaryAction `json:"generate_summary,omitempty"`
}

type BeforeSummarizeAction struct {
	// Messages is the original state messages before summarization.
	Messages []adk.Message `json:"messages,omitempty"`
}

type AfterSummarizeAction struct {
	// Messages is the final state messages after summarization.
	Messages []adk.Message `json:"messages,omitempty"`
}

type GenerateSummaryAction struct {
	// IsRetry indicates whether the generate attempt is a retry.
	IsRetry bool `json:"is_retry,omitempty"`

	// IsFailover indicates whether the generate attempt is a failover.
	IsFailover bool `json:"is_failover,omitempty"`

	// ModelResponse is the raw response returned by the model, if any.
	ModelResponse adk.Message `json:"model_response,omitempty"`
}
