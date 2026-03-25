package automemory

import "fmt"

const (
	defaultMemoryInstruction = `# auto memory

You have a persistent auto memory directory at "{memory_dir}". Its contents persist across conversations.

As you work, consult your memory files to build on previous experience.

## How to save memories:
- Organize memory semantically by topic, not chronologically
- Use the Write and Edit tools to update your memory files
- 'MEMORY.md' is always loaded into your conversation context — content is truncated after 200 lines or 4KB, so keep it concise
- Create separate topic files (e.g., 'debugging.md'', 'patterns.md'') for detailed notes and link to them from MEMORY.md
- Update or remove memories that turn out to be wrong or outdated
- Do not write duplicate memories. First check if there is an existing memory you can update before writing a new one.

## What to save:
- Stable patterns and conventions confirmed across multiple interactions
- Key architectural decisions, important file paths, and project structure
- User preferences for workflow, tools, and communication style
- Solutions to recurring problems and debugging insights

## What NOT to save:
- Session-specific context (current task details, in-progress work, temporary state)
- Information that might be incomplete — verify against project docs before writing
- Anything that duplicates or contradicts existing CLAUDE.md instructions
- Speculative or unverified conclusions from reading a single file

## Explicit user requests:
- When the user asks you to remember something across sessions (e.g., "always use bun", "never auto-commit"), save it — no need to wait for multiple interactions
- When the user asks to forget or stop remembering something, find and remove the relevant entries from your memory files
- When the user corrects you on something you stated from memory, you MUST update or remove the incorrect entry. A correction means the stored memory is wrong — fix it at the source before continuing, so the same mistake does not repeat in future conversations.

## Searching past context
- Search topic files in your memory directory: Grep with pattern="<search term>" path="{memory_dir}" glob="*.md"
- Use narrow search terms (error messages, file paths, function names) rather than broad keywords.

`

	defaultAppendCurrentIndexTruncNotify = `WARNING: MEMORY.md was truncated (lines: {memory_lines}, limit: 200; byte limit: 4096). Move detailed content into separate topic files and keep MEMORY.md as a concise index.`

	defaultAppendEmptyIndexTemplate = `Your MEMORY.md is currently empty. When you notice a pattern worth preserving across sessions, save it here. Anything in MEMORY.md will be included in your system prompt next time.`

	defaultTopicSelectionSystemPrompt = `You are selecting memories that will be useful to Claude Code as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of RELATIVE FILE PATHS (relative to the memory directory) for the memories that will clearly be useful to Claude Code as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name/description/type.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools (Claude Code is already exercising them). DO still select memories containing warnings, gotchas, or known issues about those tools — active use is exactly when those matter.`

	defaultTopicSelectionUserPrompt = `Query: {user_query}

Available memories:
{available_memories}

Recently used tools:
{tools}`

	defaultTopicMemoryTruncNotify = `
> This memory file was truncated ({reason}). Use the Read tool to view the complete file at: {abs_path}`
)

func buildExtractAutoOnlyPrompt(memoryDir string, newMessageCount int, existingMemories string, skipIndex bool) string {
	manifest := ""
	if existingMemories != "" {
		manifest = fmt.Sprintf("\n\n## Existing memory files\n\n%s\n\nCheck this list before writing — update an existing file rather than creating a duplicate.", existingMemories)
	}

	howToSave := []string{
		"## How to save memories",
		"",
		"Saving a memory is a two-step process:",
		"",
		"Step 1 — write the memory to its own file.",
		"Step 2 — add a pointer to that file in MEMORY.md. MEMORY.md is an index, not the memory body.",
		"",
		"- Keep MEMORY.md concise because it is loaded into system prompt context.",
		"- Organize memory semantically by topic, not chronologically.",
		"- Update or remove memories that turn out to be wrong or outdated.",
		"- Do not write duplicate memories.",
	}
	if skipIndex {
		howToSave = []string{
			"## How to save memories",
			"",
			"Write each memory to its own file. Do not create duplicate files.",
		}
	}

	parts := []string{
		fmt.Sprintf("You are now acting as the memory extraction subagent. Analyze only the most recent ~%d messages above and use them to update persistent memory.", newMessageCount),
		"",
		fmt.Sprintf("Memory directory: %s", memoryDir),
		"",
		"Available tools: read_file, glob, write_file, edit_file. Only paths inside the memory directory are allowed. All other tools are denied.",
		"",
		"You have a limited turn budget. read_file should happen first for every file you may update, then write_file/edit_file should happen after that. Do not interleave read and write across many turns.",
		"",
		fmt.Sprintf("You MUST only use content from the last ~%d messages to update memories. Do not investigate code or verify against source files further.", newMessageCount) + manifest,
		"",
		"If the user explicitly asks you to remember something, save it immediately. If they ask you to forget something, find and remove the relevant memory.",
		"",
		"## What to save",
		"- Stable patterns and conventions confirmed across multiple interactions",
		"- Important file paths, architectural decisions, and user preferences",
		"- Recurring debugging insights and known gotchas",
		"",
		"## What NOT to save",
		"- Session-specific temporary state or current task details",
		"- Secrets, credentials, or personal data",
		"- Speculative or unverified conclusions",
		"",
	}
	parts = append(parts, howToSave...)
	return joinLines(parts)
}

func joinLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	out := lines[0]
	for i := 1; i < len(lines); i++ {
		out += "\n" + lines[i]
	}
	return out
}
