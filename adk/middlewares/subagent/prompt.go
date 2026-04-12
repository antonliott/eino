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

// Package subagent provides a ChatModelAgentMiddleware that injects Agent, TaskOutput,
// and TaskStop tools for spawning and managing sub-agents.
package subagent

// This file contains prompt templates and tool descriptions adapted from ClaudeCode.
// Original source: https://claude.com/product/claude-code
//
// These prompts are used under the terms of the original project's open source license.

const (
	agentToolPrompt = `
# 'agent' (subagent spawner)

You have access to an 'agent' tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.
You should proactively use the 'agent' tool with specialized agents when the task at hand matches the agent's description.

When to use the agent tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the agent tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the agent tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Agent Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for agents. Whenever you have independent steps to complete - make tool_calls, or kick off agents (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the 'agent' tool to silo independent tasks within a multi-part objective.
- You should use the 'agent' tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.
`

	agentToolPromptChinese = `
# 'agent'（子代理生成器）

你可以使用 'agent' 工具来启动处理独立任务的短期子代理。这些代理是临时的——它们只在任务持续期间存在，并返回单个结果。
当手头的任务与代理的描述匹配时，你应该主动使用带有专门代理的 'agent' 工具。

何时使用 agent 工具：
- 当任务复杂且包含多个步骤，并且可以完全独立委托时
- 当任务独立于其他任务并且可以并行运行时
- 当任务需要集中推理或大量 token/上下文使用，这会使编排器线程膨胀时
- 当沙箱化提高可靠性时（例如代码执行、结构化搜索、数据格式化）
- 当你只关心子代理的输出，而不关心中间步骤时（例如执行大量研究然后返回综合报告，执行一系列计算或查找以获得简洁、相关的答案）

子代理生命周期：
1. **生成** → 提供清晰的角色、指令和预期输出
2. **运行** → 子代理自主完成任务
3. **返回** → 子代理提供单个结构化结果
4. **协调** → 将结果合并或综合到主线程中

何时不使用 agent 工具：
- 如果你需要在子代理完成后查看中间推理或步骤（agent 工具会隐藏它们）
- 如果任务很简单（几个工具调用或简单查找）
- 如果委托不会减少 token 使用、复杂性或上下文切换
- 如果拆分会增加延迟而没有好处

## 重要的 Agent 工具使用注意事项
- 尽可能并行化你所做的工作。这对于 tool_calls 和 agents 都适用。每当你有独立的步骤要完成时——进行 tool_calls，或并行启动代理（子代理）以更快地完成它们。这为用户节省了时间，这非常重要。
- 记住使用 'agent' 工具在多部分目标中隔离独立任务。
- 每当你有一个需要多个步骤的复杂任务，并且独立于代理需要完成的其他任务时，你应该使用 'agent' 工具。这些代理非常有能力且高效。
`

	agentToolDescription = `Launch a new agent to handle complex, multi-step tasks autonomously.

The Agent tool launches specialized agents (subprocesses) that autonomously handle complex tasks. Each agent type has specific capabilities and tools available to it.

Available agent types and the tools they have access to:
{other_agents}

When using the Agent tool, you must specify a subagent_type parameter to select which agent type to use.

When NOT to use the Agent tool:
- If you want to read a specific file path, use the Read or Glob tool instead of the Agent tool, to find the match more quickly
- If you are searching for a specific class definition like "class Foo", use the Glob tool instead, to find the match more quickly
- If you are searching for code within a specific file or set of 2-3 files, use the Read tool instead of the Agent tool, to find the match more quickly
- Other tasks that are not related to the agent descriptions above


Usage notes:
- Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
- When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
- Provide clear, detailed prompts so the agent can work autonomously and return exactly the information you need.
- The agent's outputs should generally be trusted
- Clearly tell the agent whether you expect it to write code or just to do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
- If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
- If the user specifies that they want you to run agents "in parallel", you MUST send a single message with multiple Agent tool use content blocks. For example, if you need to launch both a code-reviewer agent and a test-runner agent in parallel, send a single message with both tool calls.
`

	agentToolDescriptionChinese = `启动新代理以自主处理复杂的多步骤任务。

Agent 工具启动专门的代理（子进程），自主处理复杂任务。每种代理类型都有特定的功能和可用的工具。

可用的代理类型及其可访问的工具：
{other_agents}

使用 Agent 工具时，你必须指定 subagent_type 参数来选择要使用的代理类型。

何时不使用 Agent 工具：
- 如果你想读取特定的文件路径，请使用 Read 或 Glob 工具而不是 Agent 工具，以更快地找到匹配项
- 如果你正在搜索特定的类定义，如"class Foo"，请使用 Glob 工具，以更快地找到匹配项
- 如果你正在特定文件或 2-3 个文件集中搜索代码，请使用 Read 工具而不是 Agent 工具，以更快地找到匹配项
- 与上述代理描述无关的其他任务


使用说明：
- 尽可能同时启动多个代理，以最大化性能；为此，使用包含多个工具使用的单条消息
- 当代理完成时，它将向你返回一条消息。代理返回的结果对用户不可见。要向用户显示结果，你应该向用户发送一条文本消息，简要总结结果。
- 提供清晰、详细的提示，以便代理可以自主工作并返回你需要的确切信息。
- 代理的输出通常应该被信任
- 明确告诉代理你期望它编写代码还是只是进行研究（搜索、文件读取、网络获取等），因为它不知道用户的意图
- 如果代理描述提到应该主动使用它，那么你应该尽力使用它即使用户没有这样要求。使用你的判断。
- 如果用户指定他们希望你"并行"运行代理，你必须发送一条包含多个 Agent 工具使用内容块的消息。例如，如果你需要并行启动代码审查代理和测试运行代理，请发送一条包含两个工具调用的消息。
`

	agentToolBackgroundPrompt = `
## Background Task Support
- You can optionally run agents in the background using the run_in_background parameter.
  When an agent runs in the background, you will be notified via the task notification channel when it completes.
- Foreground vs background: Use foreground (default) when you need the agent's results before
  you can proceed. Use background when you have genuinely independent work to do in parallel.
- Use task_output tool to check background task status. Use task_stop tool to cancel a background task.
`

	agentToolBackgroundPromptChinese = `
## 后台任务支持
- 你可以使用 run_in_background 参数在后台运行代理。
  当代理在后台运行时，你将通过任务通知通道在它完成时收到通知。
- 前台与后台：当你需要代理结果才能继续时使用前台（默认）。当你有真正独立的工作可以并行完成时使用后台。
- 使用 task_output 工具检查后台任务状态。使用 task_stop 工具取消后台任务。
`

	taskOutputToolDescription = `Retrieve output from a running or completed background task.

- Takes a task_id parameter identifying the task
- Returns the task entry with status, result, and error information
- Use this tool when you need to check the status of a background task
`

	taskOutputToolDescriptionChinese = `获取正在运行或已完成的后台任务的输出。

- 接受 task_id 参数来标识任务
- 返回包含状态、结果和错误信息的任务条目
- 当你需要检查后台任务的状态时使用此工具
`

	taskStopToolDescription = `Stop a running background task by its ID.

- Takes a task_id parameter identifying the task to stop
- Returns a success or failure status
- Use this tool when you need to cancel a long-running background task
`

	taskStopToolDescriptionChinese = `通过 ID 停止正在运行的后台任务。

- 接受 task_id 参数来标识要停止的任务
- 返回成功或失败状态
- 当你需要取消一个长时间运行的后台任务时使用此工具
`
)
