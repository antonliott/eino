package automemory

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/slongfield/pyfmt"
	"gopkg.in/yaml.v3"

	"github.com/cloudwego/eino/adk"
	adkfs "github.com/cloudwego/eino/adk/middlewares/filesystem"
	fsmw "github.com/cloudwego/eino/adk/middlewares/filesystem"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

func init() {
	schema.RegisterName[*memoryExtra]("_eino_adk_automemory_extra")
}

type Config struct {
	MemoryDirectory string

	MemoryBackend Backend

	// Read controls how memories are loaded and injected.
	// Optional. Defaults to Sync load with topic selection enabled (if Model is set).
	Read *ReadConfig

	// Model is the default tool-calling model used by topic selection and memory extraction.
	// Per-read/per-write overrides can be configured in Read.Model / Write.Model.
	Model model.ToolCallingChatModel

	// Write controls post-run memory extraction and persistence.
	// Optional. Default: disabled.
	Write *WriteConfig

	// Coordination controls session identity and distributed async extraction coordination.
	// Optional. Defaults to a local in-process coordinator.
	Coordination *CoordinationConfig

	// OnError is called when automemory encounters an error. Errors are best-effort by default:
	// the middleware will skip memory injection and allow the agent to continue.
	// Optional.
	OnError func(ctx context.Context, stage string, err error)
}

type ReadMode string

const (
	ReadModeSync  ReadMode = "sync"
	ReadModeAsync ReadMode = "async"
)

type ReadConfig struct {
	Mode ReadMode

	// Model is used for topic selection. Defaults to Config.Model.
	Model model.ToolCallingChatModel

	// Instruction overrides the default auto memory instruction block appended to system prompt.
	// Optional.
	Instruction *string

	// Index controls how MEMORY.md is loaded into system prompt.
	// Optional.
	Index *IndexConfig

	// TopicSelection controls the "LLM select topics" path.
	// Optional. If nil, topic selection is disabled.
	TopicSelection *TopicSelectionConfig
}

type IndexConfig struct {
	FileName string
	MaxLines int
	MaxBytes int
}

type TopicSelectionConfig struct {
	// CandidateGlob is matched against the RELATIVE path under MemoryDirectory.
	// Example: "**/*.md"
	CandidateGlob  string
	CandidateLimit int
	// CandidatePreviewLines are read from each candidate to parse YAML frontmatter.
	CandidatePreviewLines int

	TopK int

	MaxLines int
	MaxBytes int
}

type WriteMode string

const (
	WriteModeDisabled WriteMode = "disabled"
	WriteModeAsync    WriteMode = "async"
	WriteModeSync     WriteMode = "sync"
)

type WriteConfig struct {
	Mode WriteMode

	// Model is used for memory extraction. Defaults to Config.Model.
	Model model.ToolCallingChatModel

	// Backend is used for persistence during extraction.
	Backend Backend

	// RecentMessages limits how many recent messages are provided to the extractor.
	RecentMessages int

	// MaxTurns caps the extractor's tool-call loop.
	MaxTurns int

	// SkipIndex mirrors Claude Code's "skipIndex" mode.
	SkipIndex bool

	// HandleExtractionIterator, if set, is called with the extractionAgent's event
	// iterator returned by Run(). The handler is responsible for draining the
	// iterator (calling Next until it returns ok=false) and returning any error
	// it wants to surface to the middleware.
	//
	// If nil, automemory uses the default drain behavior: ignore all events and
	// return the first ev.Err encountered (if any).
	HandleExtractionIterator func(ctx context.Context, iter *adk.AsyncIterator[*adk.AgentEvent]) error
}

type middleware struct {
	adk.BaseChatModelAgentMiddleware

	cfg *Config

	topicSelectionModel model.ToolCallingChatModel
	extractionAgent     *adk.ChatModelAgent
	topicSelectionTool  *schema.ToolInfo
	coordination        *CoordinationConfig
}

type selectionFuture struct {
	done chan struct{}
	mu   sync.Mutex

	// Store an immutable snapshot to avoid being mutated via shared pointers.
	content string
	err     error
	applied bool
}

type ctxKeySelectionFuture struct{}

const (
	memoryExtraKey    = "__eino_automemory__"
	instructionMarker = "<!-- automemory:instruction -->"
)

type memoryExtra struct {
	Type       string
	Cursor     int
	UpdatedAt  string
	Visibility string
	SchemaVer  int
}

func New(ctx context.Context, config *Config) (adk.ChatModelAgentMiddleware, error) {
	if config == nil || config.MemoryDirectory == "" || config.MemoryBackend == nil {
		return nil, fmt.Errorf("auto memory config: invalid")
	}
	if config.Read == nil {
		config.Read = &ReadConfig{}
	}
	applyReadDefaults(config)

	m := &middleware{
		BaseChatModelAgentMiddleware: adk.BaseChatModelAgentMiddleware{},
		cfg:                          config,
		coordination:                 config.Coordination,
	}

	m.topicSelectionTool = topicSelectionToolInfo()
	if config.Read.TopicSelection != nil && config.Read.Model != nil {
		bound, err := config.Read.Model.WithTools([]*schema.ToolInfo{m.topicSelectionTool})
		if err != nil {
			return nil, fmt.Errorf("auto memory topic selection model init failed: %w", err)
		}
		m.topicSelectionModel = bound
	}

	writeFSBackend, err := newFSBackend(config.Write.Backend, config.MemoryDirectory)
	if err != nil {
		return nil, err
	}
	fileSystemMiddleware, err := fsmw.New(ctx, &fsmw.MiddlewareConfig{
		Backend:        writeFSBackend,
		LsToolConfig:   &fsmw.ToolConfig{Disable: true},
		GrepToolConfig: &fsmw.ToolConfig{Disable: true},
	})
	if err != nil {
		return nil, err
	}

	if config.Write.Mode != WriteModeDisabled && config.Write.Model != nil {
		agent, err := adk.NewChatModelAgent(ctx, &adk.ChatModelAgentConfig{
			Name:          "automemory_extractor",
			Description:   "Internal auto memory extraction subagent",
			Model:         config.Write.Model,
			Handlers:      []adk.ChatModelAgentMiddleware{fileSystemMiddleware},
			MaxIterations: config.Write.MaxTurns,
		})
		if err != nil {
			return nil, fmt.Errorf("auto memory extraction agent init failed: %w", err)
		}
		m.extractionAgent = agent
	}

	return m, nil
}

func (m *middleware) BeforeAgent(ctx context.Context, runCtx *adk.ChatModelAgentContext) (context.Context, *adk.ChatModelAgentContext, error) {
	if runCtx == nil {
		return ctx, runCtx, nil
	}
	nRunCtx := *runCtx

	// Sync distributed write cursor back into message extras so later runs on other
	// machines still carry a transcript-local marker.
	if nRunCtx.AgentInput != nil && len(nRunCtx.AgentInput.Messages) > 0 && m.coordination != nil && m.coordination.Coordinator != nil {
		if sessionID, err := m.resolveSessionID(ctx, &adk.ChatModelAgentState{Messages: nRunCtx.AgentInput.Messages}); err == nil && sessionID != "" {
			localCursor := getWriteCursorFromMessages(nRunCtx.AgentInput.Messages)
			if remoteCursor, ok, err := m.coordination.Coordinator.GetCursor(ctx, sessionID); err == nil && ok && remoteCursor > localCursor {
				st := markWriteCursor(&adk.ChatModelAgentState{Messages: nRunCtx.AgentInput.Messages}, remoteCursor)
				if st != nil {
					nRunCtx.AgentInput = &adk.AgentInput{
						Messages:        st.Messages,
						EnableStreaming: nRunCtx.AgentInput.EnableStreaming,
					}
				}
			}
		}
	}

	// If automemory was already injected into the instruction or message list,
	// skip all memory-loading work for this run and let the agent continue.
	if hasInstructionInjected(nRunCtx.Instruction) || (nRunCtx.AgentInput != nil && alreadyInjected(nRunCtx.AgentInput.Messages)) {
		return ctx, &nRunCtx, nil
	}

	// 1) System prompt: inject auto memory instruction + MEMORY.md content (best-effort).
	nRunCtx.Instruction = m.injectIndexIntoInstruction(ctx, nRunCtx.Instruction)

	// 2) Topic memories: sync mode injects before the user's query.
	if m.cfg.Read.Mode == ReadModeSync && m.cfg.Read.TopicSelection != nil && m.topicSelectionModel != nil {
		memMsg, err := m.selectAndBuildTopicMemoryMessage(ctx, nRunCtx.AgentInput)
		if err != nil {
			m.onErr(ctx, OnErrorStageTopicSelectionSync, err)
		} else if memMsg != nil && nRunCtx.AgentInput != nil && len(nRunCtx.AgentInput.Messages) > 0 {
			msgs := append([]adk.Message{}, nRunCtx.AgentInput.Messages...)
			msgs = append(msgs, memMsg)
			nRunCtx.AgentInput = &adk.AgentInput{Messages: msgs, EnableStreaming: nRunCtx.AgentInput.EnableStreaming}
		}
	}

	// 3) Topic memories: async mode starts selection here (cannot use RunLocalValue in BeforeAgent).
	if m.cfg.Read.Mode == ReadModeAsync && m.cfg.Read.TopicSelection != nil && m.topicSelectionModel != nil && nRunCtx.AgentInput != nil {
		if existing, _ := ctx.Value(ctxKeySelectionFuture{}).(*selectionFuture); existing == nil {
			fut := &selectionFuture{done: make(chan struct{})}
			ctx = context.WithValue(ctx, ctxKeySelectionFuture{}, fut)

			// Snapshot current messages for selection; async path is best-effort.
			msgSnapshot := append([]adk.Message{}, nRunCtx.AgentInput.Messages...)
			go func() {
				defer close(fut.done)
				memMsg, selErr := m.selectAndBuildTopicMemoryMessage(ctx, &adk.AgentInput{Messages: msgSnapshot})
				fut.mu.Lock()
				defer fut.mu.Unlock()
				if selErr != nil {
					fut.err = selErr
					return
				}
				if memMsg != nil {
					fut.content = memMsg.Content
				}
			}()
		}
	}

	return ctx, &nRunCtx, nil
}

func (m *middleware) BeforeModelRewriteState(ctx context.Context, state *adk.ChatModelAgentState, _ *adk.ModelContext) (context.Context, *adk.ChatModelAgentState, error) {
	if state == nil {
		return ctx, state, nil
	}
	// Best-effort protection: if automemory content has been injected before and later
	// mutated by other components, restore it using the immutable snapshot stored in the future.
	if fut, _ := ctx.Value(ctxKeySelectionFuture{}).(*selectionFuture); fut != nil {
		fut.mu.Lock()
		expected := fut.content
		fut.mu.Unlock()
		if strings.TrimSpace(expected) != "" {
			state = ensureMemoryMsgUnchanged(state, expected)
		}
	}
	if m.cfg.Read.Mode != ReadModeAsync {
		return ctx, state, nil
	}
	fut, _ := ctx.Value(ctxKeySelectionFuture{}).(*selectionFuture)
	if fut == nil {
		return ctx, state, nil
	}

	select {
	case <-fut.done:
	default:
		return ctx, state, nil
	}

	fut.mu.Lock()
	if fut.applied {
		fut.mu.Unlock()
		return ctx, state, nil
	}
	content := fut.content
	err := fut.err
	fut.mu.Unlock()
	if err != nil {
		m.onErr(ctx, OnErrorStageTopicSelectionAsync, err)
	}

	var msgs []adk.Message
	if strings.TrimSpace(content) != "" {
		msgs = append(msgs, state.Messages...)
		msgs = append(msgs, newMemoryMessage(content))
	} else {
		msgs = state.Messages
	}

	fut.mu.Lock()
	fut.applied = true
	fut.mu.Unlock()

	return ctx, &adk.ChatModelAgentState{Messages: msgs}, nil
}

func applyReadDefaults(cfg *Config) {
	if cfg.Read.Mode == "" {
		cfg.Read.Mode = ReadModeSync
	}
	if cfg.Read.Index == nil {
		cfg.Read.Index = &IndexConfig{}
	}
	if cfg.Read.Index.FileName == "" {
		cfg.Read.Index.FileName = memoryIndexFileName
	}
	if cfg.Read.Index.MaxLines <= 0 {
		cfg.Read.Index.MaxLines = defaultIndexMaxLines
	}
	if cfg.Read.Index.MaxBytes <= 0 {
		cfg.Read.Index.MaxBytes = defaultIndexMaxBytes
	}

	if cfg.Read.Model == nil {
		cfg.Read.Model = cfg.Model
	}

	// Default topic selection on if a selector model is provided.
	if cfg.Read.TopicSelection == nil && cfg.Read.Model != nil {
		cfg.Read.TopicSelection = &TopicSelectionConfig{}
	}
	if cfg.Read.TopicSelection != nil {
		if cfg.Read.TopicSelection.TopK <= 0 {
			cfg.Read.TopicSelection.TopK = defaultTopicTopK
		}
		if cfg.Read.TopicSelection.CandidateGlob == "" {
			cfg.Read.TopicSelection.CandidateGlob = CandidateGlobPattern
		}
		if cfg.Read.TopicSelection.CandidateLimit <= 0 {
			cfg.Read.TopicSelection.CandidateLimit = defaultCandidateLimit
		}
		if cfg.Read.TopicSelection.CandidatePreviewLines <= 0 {
			cfg.Read.TopicSelection.CandidatePreviewLines = defaultCandidatePreviewLine
		}
		if cfg.Read.TopicSelection.MaxLines <= 0 {
			cfg.Read.TopicSelection.MaxLines = defaultTopicMaxLines
		}
		if cfg.Read.TopicSelection.MaxBytes <= 0 {
			cfg.Read.TopicSelection.MaxBytes = defaultTopicMaxBytes
		}
	}

	if cfg.Write == nil {
		cfg.Write = &WriteConfig{Mode: WriteModeDisabled}
	}
	if cfg.Write.Mode == "" {
		cfg.Write.Mode = WriteModeDisabled
	}
	if cfg.Write.Model == nil {
		cfg.Write.Model = cfg.Model
	}
	if cfg.Write.Backend == nil {
		// Backend now includes write/edit capabilities.
		cfg.Write.Backend = cfg.MemoryBackend
	}
	if cfg.Write.RecentMessages <= 0 {
		cfg.Write.RecentMessages = 20
	}
	if cfg.Write.MaxTurns <= 0 {
		cfg.Write.MaxTurns = defaultMemoryWriteMaxTurns
	}

	if cfg.Coordination == nil {
		cfg.Coordination = &CoordinationConfig{}
	}
	if cfg.Coordination.Coordinator == nil {
		cfg.Coordination.Coordinator = NewLocalCoordinator()
	}
	if cfg.Coordination.LockTTL <= 0 {
		cfg.Coordination.LockTTL = 2 * time.Minute
	}
}

type topicSelectionResp struct {
	SelectedMemories []string `json:"selected_memories"`
}

func (m *middleware) injectIndexIntoInstruction(ctx context.Context, baseInstruction string) string {
	memDir := m.cfg.MemoryDirectory

	var memDesc string
	if m.cfg.Read.Instruction != nil {
		memDesc = *m.cfg.Read.Instruction
	} else {
		s, err := pyfmt.Fmt(defaultMemoryInstruction, map[string]any{"memory_dir": memDir})
		if err != nil {
			m.onErr(ctx, OnErrorStageRenderInstruction, err)
			return baseInstruction
		}
		memDesc = s
	}

	indexPath := filepath.Join(memDir, m.cfg.Read.Index.FileName)
	indexContent := ""
	totalLines := 0

	fc, err := m.cfg.MemoryBackend.Read(ctx, &ReadRequest{FilePath: indexPath})
	if err == nil && fc != nil {
		indexContent = fc.Content
		totalLines = strings.Count(indexContent, "\n") + 1
	} else {
		// Missing index is not fatal; keep empty.
		indexContent = ""
	}

	sb := make([]string, 0, 5)
	sb = append(sb, memDesc)
	sb = append(sb, "## "+m.cfg.Read.Index.FileName)
	if strings.TrimSpace(indexContent) == "" {
		sb = append(sb, defaultAppendEmptyIndexTemplate)
	} else {
		truncatedMemoryIndex, _, truncated := linesOrSizeTrunc(indexContent, m.cfg.Read.Index.MaxLines, m.cfg.Read.Index.MaxBytes)
		sb = append(sb, truncatedMemoryIndex)
		if truncated {
			notify, err := pyfmt.Fmt(defaultAppendCurrentIndexTruncNotify, map[string]any{
				"memory_lines": totalLines,
			})
			if err == nil {
				sb = append(sb, notify)
			}
		}
	}

	return baseInstruction + "\n" + instructionMarker + "\n" + strings.Join(sb, "\n")
}

func linesOrSizeTrunc(content string, lines, size int) (newContent string, reason string, truncated bool) {
	linesTrunc := func(content string, lines int) {
		sp := strings.Split(content, "\n")
		if len(sp) > lines {
			newContent = strings.Join(sp[:lines], "\n")
			reason = fmt.Sprintf("first %d lines", lines)
			truncated = true
		} else {
			newContent = content
		}
	}

	sizeTrunc := func(content string, size int) {
		if len(content) > size {
			newContent = content[:size]
			reason = fmt.Sprintf("%d byte limit", size)
			truncated = true
		} else {
			newContent = content
		}
	}

	if lines == 0 && size == 0 {
		return content, "", false
	} else if lines == 0 {
		sizeTrunc(content, size)
	} else if size == 0 {
		linesTrunc(content, lines)
	} else {
		linesTrunc(content, lines)
		sizeTrunc(newContent, size)
	}
	return
}

func (m *middleware) onErr(ctx context.Context, stage string, err error) {
	if err == nil {
		return
	}
	if m.cfg != nil && m.cfg.OnError != nil {
		m.cfg.OnError(ctx, stage, err)
	}
}

type topicFrontmatter struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description"`
	Type        string `yaml:"type"`
}

func parseFrontmatter(md string) (fm topicFrontmatter, ok bool) {
	// Only consider YAML frontmatter at the beginning.
	s := strings.TrimLeft(md, "\ufeff \t\r\n")
	if !strings.HasPrefix(s, "---\n") && !strings.HasPrefix(s, "---\r\n") {
		return topicFrontmatter{}, false
	}
	// Find the next delimiter.
	parts := strings.SplitN(s, "\n---", 2)
	if len(parts) != 2 {
		return topicFrontmatter{}, false
	}
	yml := strings.TrimPrefix(parts[0], "---\n")
	if err := yaml.Unmarshal([]byte(yml), &fm); err != nil {
		return topicFrontmatter{}, false
	}
	return fm, true
}

func (m *middleware) selectAndBuildTopicMemoryMessage(ctx context.Context, agentIn *adk.AgentInput) (*schema.Message, error) {
	if agentIn == nil || len(agentIn.Messages) == 0 {
		return nil, nil
	}
	if m.cfg.Read.TopicSelection == nil || m.topicSelectionModel == nil {
		return nil, nil
	}

	last := agentIn.Messages[len(agentIn.Messages)-1]
	if last == nil || last.Role != schema.User {
		return nil, nil
	}

	// 1) List candidate topic files.
	files, globErr := m.cfg.MemoryBackend.GlobInfo(ctx, &GlobInfoRequest{
		Pattern: m.cfg.Read.TopicSelection.CandidateGlob,
		Path:    m.cfg.MemoryDirectory,
	})
	if globErr != nil {
		return nil, globErr
	}
	if len(files) == 0 {
		return nil, nil
	}

	indexAbs := filepath.Join(m.cfg.MemoryDirectory, m.cfg.Read.Index.FileName)
	candidates := make([]FileInfo, 0, len(files))
	for _, fi := range files {
		if filepath.Clean(fi.Path) == filepath.Clean(indexAbs) {
			continue
		}
		candidates = append(candidates, fi)
	}
	if len(candidates) == 0 {
		return nil, nil
	}

	// Sort by modified time (desc) and cap.
	sort.Slice(candidates, func(i, j int) bool {
		return parseRFC3339NanoBestEffort(candidates[i].ModifiedAt).After(parseRFC3339NanoBestEffort(candidates[j].ModifiedAt))
	})
	if len(candidates) > m.cfg.Read.TopicSelection.CandidateLimit {
		candidates = candidates[:m.cfg.Read.TopicSelection.CandidateLimit]
	}

	// 2) Build "available memories" manifest.
	type bundle struct {
		AbsPath string
		RelPath string
		Info    FileInfo
	}
	relToAbs := make(map[string]bundle, len(candidates))
	available := make([]string, 0, len(candidates))
	orderedRel := make([]string, 0, len(candidates))

	for _, fi := range candidates {
		rel, relErr := filepath.Rel(m.cfg.MemoryDirectory, fi.Path)
		if relErr != nil {
			rel = filepath.Base(fi.Path)
		}
		rel = filepath.ToSlash(rel)

		preview, err := m.cfg.MemoryBackend.Read(ctx, &ReadRequest{
			FilePath: fi.Path,
			Limit:    m.cfg.Read.TopicSelection.CandidatePreviewLines,
		})
		if err != nil {
			// best-effort: skip this candidate
			continue
		}
		desc := ""
		if fm, ok := parseFrontmatter(preview.Content); ok {
			if strings.TrimSpace(fm.Description) != "" {
				desc = strings.TrimSpace(fm.Description)
			} else if strings.TrimSpace(fm.Name) != "" {
				desc = strings.TrimSpace(fm.Name)
			}
			if strings.TrimSpace(fm.Type) != "" {
				if desc == "" {
					desc = "type=" + strings.TrimSpace(fm.Type)
				} else {
					desc = desc + " (type=" + strings.TrimSpace(fm.Type) + ")"
				}
			}
		}
		if desc == "" {
			snippet, _, _ := linesOrSizeTrunc(preview.Content, 3, 256)
			desc = strings.TrimSpace(snippet)
		}

		available = append(available, fmt.Sprintf("- %s (saved %s): %s", rel, fi.ModifiedAt, desc))
		relToAbs[rel] = bundle{AbsPath: fi.Path, RelPath: rel, Info: fi}
		orderedRel = append(orderedRel, rel)
	}

	topK := m.cfg.Read.TopicSelection.TopK
	if topK <= 0 {
		topK = defaultTopicTopK
	}

	var selected []string

	// 3) Fast path: if candidates <= topK, skip model selection and surface all.
	if len(orderedRel) <= topK {
		selected = orderedRel
	} else {
		// 4) Recently used tools from the current run messages.
		dedupTools := make(map[string]struct{})
		for _, msg := range agentIn.Messages {
			if msg != nil && msg.Role == schema.Tool && msg.ToolName != "" {
				dedupTools[msg.ToolName] = struct{}{}
			}
		}
		tools := make([]string, 0, len(dedupTools))
		for t := range dedupTools {
			tools = append(tools, t)
		}
		sort.Strings(tools)

		userMsg, err := pyfmt.Fmt(defaultTopicSelectionUserPrompt, map[string]any{
			"user_query":         last.Content,
			"available_memories": strings.Join(available, "\n"),
			"tools":              strings.Join(tools, ", "),
		})
		if err != nil {
			return nil, err
		}

		toolInfo := topicSelectionToolInfo()
		resp, err := m.topicSelectionModel.Generate(
			ctx,
			[]*schema.Message{
				schema.SystemMessage(defaultTopicSelectionSystemPrompt),
				schema.UserMessage(userMsg),
			},
			model.WithToolChoice(schema.ToolChoiceForced, toolInfo.Name),
		)
		if err != nil {
			return nil, err
		}

		// Prefer parsing tool call arguments (structured).
		valid := make(map[string]struct{}, len(relToAbs))
		for k := range relToAbs {
			valid[k] = struct{}{}
		}
		selected, err = parseTopicSelectionFromToolCall(resp, valid)
		if err != nil {
			return nil, err
		}
		if len(selected) == 0 {
			return nil, nil
		}
	}

	// 5) Read selected topics (truncate) and return as a meta user message.
	var rendered []string
	for _, rel := range selected {
		if len(rendered) >= topK {
			break
		}
		b, ok := relToAbs[rel]
		if !ok {
			// Ignore unknown selections (best-effort).
			continue
		}
		full, err := m.cfg.MemoryBackend.Read(ctx, &ReadRequest{FilePath: b.AbsPath})
		if err != nil {
			continue
		}

		content, truncReason, truncated := linesOrSizeTrunc(full.Content, m.cfg.Read.TopicSelection.MaxLines, m.cfg.Read.TopicSelection.MaxBytes)
		if truncated {
			truncNotify, err := pyfmt.Fmt(defaultTopicMemoryTruncNotify, map[string]any{
				"reason":   truncReason,
				"abs_path": b.AbsPath,
			})
			if err == nil {
				content += truncNotify
			}
		}
		rendered = append(rendered, fmt.Sprintf("<system-reminder>\nContents of %s (saved %s):\n\n%s\n</system-reminder>", b.AbsPath, b.Info.ModifiedAt, content))
	}
	if len(rendered) == 0 {
		return nil, nil
	}

	return newMemoryMessage("<!-- automemory -->\n" + strings.Join(rendered, "\n\n")), nil
}

func topicSelectionToolInfo() *schema.ToolInfo {
	return &schema.ToolInfo{
		Name: topicSelectionToolName,
		Desc: "Select which memory files to surface for the current query. Return selected_memories as RELATIVE paths (relative to the memory directory).",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"selected_memories": {
				Type:     schema.Array,
				Desc:     "Relative paths of selected memory files, e.g. \"debugging.md\" or \"notes/patterns.md\".",
				Required: true,
				ElemInfo: &schema.ParameterInfo{Type: schema.String},
			},
		}),
	}
}

func parseTopicSelectionFromToolCall(msg *schema.Message, valid map[string]struct{}) ([]string, error) {
	if msg == nil || len(msg.ToolCalls) == 0 {
		return nil, fmt.Errorf("no tool calls")
	}
	tc := msg.ToolCalls[0]
	if tc.Function.Name != topicSelectionToolName {
		return nil, fmt.Errorf("unexpected tool call: %s", tc.Function.Name)
	}
	var parsed topicSelectionResp
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err != nil {
		return nil, err
	}
	out := normalizeSelected(parsed.SelectedMemories)
	// Filter to known candidates to avoid hallucinated paths.
	filtered := make([]string, 0, len(out))
	for _, p := range out {
		if _, ok := valid[p]; ok {
			filtered = append(filtered, p)
		}
	}
	return filtered, nil
}

func normalizeSelected(in []string) []string {
	out := make([]string, 0, len(in))
	seen := make(map[string]struct{}, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		s = strings.TrimPrefix(s, "./")
		s = filepath.ToSlash(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func alreadyInjected(msgs []adk.Message) bool {
	for _, m := range msgs {
		if isMemoryMessage(m) {
			return true
		}
	}
	return false
}

func isMemoryMessage(m *schema.Message) bool {
	if m == nil || m.Role != schema.User {
		return false
	}
	if m.Extra != nil {
		if v, ok := m.Extra[memoryExtraKey]; ok && v != nil {
			return true
		}
	}
	// Backward compatible marker (older versions).
	return strings.Contains(m.Content, "<!-- automemory -->")
}

func hasInstructionInjected(instruction string) bool {
	return strings.Contains(instruction, instructionMarker)
}

func newMemoryMessage(content string) *schema.Message {
	msg := schema.UserMessage(content)
	if msg.Extra == nil {
		msg.Extra = map[string]any{}
	}
	msg.Extra[memoryExtraKey] = &memoryExtra{
		Type: "memory",
	}
	return msg
}

func ensureMemoryMsgUnchanged(state *adk.ChatModelAgentState, expectedContent string) *adk.ChatModelAgentState {
	if state == nil || strings.TrimSpace(expectedContent) == "" {
		return state
	}
	changed := false
	out := *state
	out.Messages = append([]adk.Message{}, state.Messages...)

	for i, m := range out.Messages {
		if !isMemoryMessage(m) {
			continue
		}
		if m.Content != expectedContent || m.Extra == nil || m.Extra[memoryExtraKey] == nil {
			out.Messages[i] = newMemoryMessage(expectedContent)
			changed = true
		}
	}
	if !changed {
		return state
	}
	return &out
}

func extractFilePath(args string) (string, bool) {
	var m map[string]any
	if err := json.Unmarshal([]byte(args), &m); err != nil {
		return "", false
	}
	if v, ok := m["file_path"]; ok {
		if s, ok := v.(string); ok && s != "" {
			return s, true
		}
	}
	if v, ok := m["filePath"]; ok { // tolerate camelCase
		if s, ok := v.(string); ok && s != "" {
			return s, true
		}
	}
	return "", false
}

func isPathWithinMemoryDir(memDir string, filePath string) bool {
	if memDir == "" || filePath == "" {
		return false
	}
	md := filepath.Clean(memDir)
	fp := filepath.Clean(filePath)
	if !filepath.IsAbs(fp) {
		fp = filepath.Join(md, fp)
		fp = filepath.Clean(fp)
	}
	if fp == md {
		return true
	}
	sep := string(filepath.Separator)
	return strings.HasPrefix(fp, md+sep)
}

func (m *middleware) AfterAgent(ctx context.Context, state *adk.ChatModelAgentState, result *adk.ChatModelAgentRunResult) (context.Context, *adk.ChatModelAgentState, error) {
	if m.cfg == nil || m.cfg.Write == nil || m.cfg.Write.Mode == WriteModeDisabled {
		return ctx, state, nil
	}
	if result != nil && (result.Interrupted || result.Err != nil) {
		return ctx, state, nil
	}
	if m.extractionAgent == nil || m.cfg.Write.Backend == nil {
		return ctx, state, nil
	}
	if state == nil || len(state.Messages) == 0 {
		return ctx, state, nil
	}

	sessionID, err := m.resolveSessionID(ctx, state)
	if err != nil {
		m.onErr(ctx, OnErrorStageResolveSessionID, err)
		return ctx, state, nil
	}

	cursor := getWriteCursorFromMessages(state.Messages)
	if sessionID != "" {
		if remoteCursor, ok, err := m.coordination.Coordinator.GetCursor(ctx, sessionID); err == nil && ok && remoteCursor > cursor {
			cursor = remoteCursor
			state = markWriteCursor(state, cursor)
		}
	}
	if cursor >= len(state.Messages) {
		return ctx, state, nil
	}

	// Skip background extraction if the main agent already wrote memory files in this range.
	if hasMemoryWritesSince(state.Messages, cursor, m.cfg.MemoryDirectory) {
		end := len(state.Messages)
		if sessionID != "" {
			_ = m.coordination.Coordinator.SetCursor(ctx, sessionID, end)
		}
		return ctx, markWriteCursor(state, end), nil
	}

	if countModelVisibleMessages(state.Messages[cursor:]) == 0 {
		end := len(state.Messages)
		if sessionID != "" {
			_ = m.coordination.Coordinator.SetCursor(ctx, sessionID, end)
		}
		return ctx, markWriteCursor(state, end), nil
	}

	switch m.cfg.Write.Mode {
	case WriteModeDisabled:
		// do nothing
		return ctx, state, nil

	case WriteModeSync:
		end := len(state.Messages)
		if err := m.runMemoryExtractionAgent(ctx, state.Messages, cursor); err != nil {
			m.onErr(ctx, OnErrorStageMemoryWriteSync, err)
			return ctx, state, nil
		}
		if sessionID != "" {
			_ = m.coordination.Coordinator.SetCursor(ctx, sessionID, end)
		}
		return ctx, markWriteCursor(state, end), nil

	case WriteModeAsync:
		if sessionID == "" {
			sessionID = getOrInitWriteSessionID(ctx)
		}
		snap, err := buildPendingSnapshot(state.Messages, cursor)
		if err != nil {
			m.onErr(ctx, OnErrorStageSnapshotMarshal, err)
			return ctx, state, nil
		}
		unlock, ok, err := m.coordination.Coordinator.AcquireLock(ctx, sessionID, m.coordination.LockTTL)
		if err != nil {
			m.onErr(ctx, OnErrorStageAcquireExtractionLock, err)
			return ctx, state, nil
		}
		if !ok {
			if err := m.coordination.Coordinator.SetPendingSnapshot(ctx, sessionID, snap); err != nil {
				m.onErr(ctx, OnErrorStageStashPendingSnapshot, err)
			}
			return ctx, state, nil
		}
		go m.runExtractionDrain(context.Background(), sessionID, unlock, snap)
		return ctx, state, nil

	default:
		return ctx, state, nil
	}
}

func getWriteCursorFromMessages(msgs []adk.Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m == nil || m.Extra == nil {
			continue
		}
		v, ok := m.Extra[memoryExtraKey]
		if !ok {
			continue
		}
		switch meta := v.(type) {
		case *memoryExtra:
			if meta != nil && meta.Type == "write_cursor" {
				return meta.Cursor
			}
		case map[string]any:
			if typ, _ := meta["type"].(string); typ != "write_cursor" {
				continue
			}
			switch c := meta["cursor"].(type) {
			case int:
				return c
			case int64:
				return int(c)
			case float64:
				return int(c)
			}
		}
	}
	return 0
}

func markWriteCursor(state *adk.ChatModelAgentState, cursor int) *adk.ChatModelAgentState {
	if state == nil || len(state.Messages) == 0 {
		return state
	}
	last := state.Messages[len(state.Messages)-1]
	if last == nil {
		return state
	}

	if last.Extra == nil {
		last.Extra = map[string]any{}
	}
	last.Extra[memoryExtraKey] = &memoryExtra{
		Type:       "write_cursor",
		Cursor:     cursor,
		UpdatedAt:  time.Now().Format(time.RFC3339Nano),
		Visibility: "internal",
		SchemaVer:  1,
	}

	return state
}

func countModelVisibleMessages(msgs []adk.Message) int {
	n := 0
	for _, m := range msgs {
		if m == nil {
			continue
		}
		if m.Role == schema.User || m.Role == schema.Assistant {
			n++
		}
	}
	return n
}

func cloneMessages(in []adk.Message) []adk.Message {
	if in == nil {
		return nil
	}
	out := make([]adk.Message, len(in))
	copy(out, in)
	return out
}

func getOrInitWriteSessionID(ctx context.Context) string {
	const key = "__automemory_write_session_id__"
	if v, ok := adk.GetSessionValue(ctx, key); ok {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	// Stable enough for in-process session identity.
	s := fmt.Sprintf("%d", time.Now().UnixNano())
	adk.AddSessionValue(ctx, key, s)
	return s
}

func (m *middleware) resolveSessionID(ctx context.Context, state *adk.ChatModelAgentState) (string, error) {
	if m.coordination != nil && m.coordination.SessionIDFunc != nil {
		return m.coordination.SessionIDFunc(ctx, state)
	}
	return getOrInitWriteSessionID(ctx), nil
}

func buildPendingSnapshot(messages []adk.Message, cursor int) (*PendingSnapshot, error) {
	raw, err := json.Marshal(messages)
	if err != nil {
		return nil, err
	}
	return &PendingSnapshot{Cursor: cursor, Messages: raw}, nil
}

func decodePendingSnapshot(snapshot *PendingSnapshot) ([]adk.Message, int, error) {
	if snapshot == nil {
		return nil, 0, nil
	}
	var msgs []adk.Message
	if err := json.Unmarshal(snapshot.Messages, &msgs); err != nil {
		return nil, 0, err
	}
	return msgs, snapshot.Cursor, nil
}

func (m *middleware) runExtractionDrain(ctx context.Context, sessionID string, unlock func(context.Context) error, initial *PendingSnapshot) {
	defer func() {
		if unlock == nil {
			return
		}
		if err := unlock(ctx); err != nil {
			m.onErr(ctx, OnErrorStageReleaseExtractionLock, err)
		}
	}()

	current := initial
	for current != nil {
		msgs, cursor, err := decodePendingSnapshot(current)
		if err != nil {
			m.onErr(ctx, OnErrorStageDecodePendingSnapshot, err)
		} else if err := m.runMemoryExtractionAgent(ctx, msgs, cursor); err != nil {
			m.onErr(ctx, OnErrorStageMemoryWriteAsync, err)
		} else {
			_ = m.coordination.Coordinator.SetCursor(ctx, sessionID, len(msgs))
		}

		next, loadErr := m.coordination.Coordinator.PopPendingSnapshot(ctx, sessionID)
		if loadErr != nil {
			m.onErr(ctx, OnErrorStageLoadPendingSnapshot, loadErr)
			return
		}
		current = next
	}
}

func hasMemoryWritesSince(msgs []adk.Message, cursor int, memoryDir string) bool {
	if cursor < 0 {
		cursor = 0
	}
	for _, msg := range msgs[cursor:] {
		if msg == nil || msg.Role != schema.Assistant {
			continue
		}
		for _, tc := range msg.ToolCalls {
			if tc.Function.Name != adkfs.ToolNameWriteFile && tc.Function.Name != adkfs.ToolNameEditFile {
				continue
			}
			if fp, ok := extractFilePath(tc.Function.Arguments); ok && isPathWithinMemoryDir(memoryDir, fp) {
				return true
			}
		}
	}
	return false
}

func countModelVisibleMessagesSince(msgs []adk.Message, cursor int) int {
	if cursor < 0 {
		cursor = 0
	}
	if cursor >= len(msgs) {
		return 0
	}
	return countModelVisibleMessages(msgs[cursor:])
}

func (m *middleware) runMemoryExtractionAgent(ctx context.Context, snapshot []adk.Message, cursor int) error {
	if len(snapshot) == 0 || cursor >= len(snapshot) {
		return nil
	}
	manifest, err := m.buildMemoryManifest(ctx)
	if err != nil {
		return err
	}
	newMessageCount := countModelVisibleMessagesSince(snapshot, cursor)
	userPrompt := buildExtractAutoOnlyPrompt(m.cfg.MemoryDirectory, newMessageCount, manifest, m.cfg.Write.SkipIndex)

	msgs := cloneMessages(snapshot)
	msgs = append(msgs, schema.UserMessage(userPrompt))

	iter := m.extractionAgent.Run(ctx, &adk.AgentInput{
		Messages:        msgs,
		EnableStreaming: false,
	})
	if m.cfg != nil && m.cfg.Write != nil && m.cfg.Write.HandleExtractionIterator != nil {
		return m.cfg.Write.HandleExtractionIterator(ctx, iter)
	}
	for {
		ev, ok := iter.Next()
		if !ok {
			return nil
		}
		if ev == nil {
			continue
		}
		if ev.Err != nil {
			return ev.Err
		}
	}
}

func (m *middleware) buildMemoryManifest(ctx context.Context) (string, error) {
	files, err := m.cfg.MemoryBackend.GlobInfo(ctx, &GlobInfoRequest{
		Pattern: CandidateGlobPattern,
		Path:    m.cfg.MemoryDirectory,
	})
	if err != nil {
		return "", err
	}
	indexAbs := filepath.Join(m.cfg.MemoryDirectory, m.cfg.Read.Index.FileName)
	lines := make([]string, 0, len(files))
	for _, fi := range files {
		rel, relErr := filepath.Rel(m.cfg.MemoryDirectory, fi.Path)
		if relErr != nil {
			rel = filepath.Base(fi.Path)
		}
		rel = filepath.ToSlash(rel)
		if filepath.Clean(fi.Path) == filepath.Clean(indexAbs) {
			rel = m.cfg.Read.Index.FileName
		}
		desc := ""
		preview, rerr := m.cfg.MemoryBackend.Read(ctx, &ReadRequest{FilePath: fi.Path, Limit: defaultCandidatePreviewLine})
		if rerr == nil && preview != nil {
			if fm, ok := parseFrontmatter(preview.Content); ok {
				desc = strings.TrimSpace(fm.Description)
			}
		}
		if desc != "" {
			lines = append(lines, fmt.Sprintf("- %s (saved %s): %s", rel, fi.ModifiedAt, desc))
		} else {
			lines = append(lines, fmt.Sprintf("- %s (saved %s)", rel, fi.ModifiedAt))
		}
	}
	return strings.Join(lines, "\n"), nil
}

func parseRFC3339NanoBestEffort(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t
	}
	return time.Time{}
}
