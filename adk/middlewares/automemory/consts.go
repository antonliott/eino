package automemory

const (
	// CandidateGlobPattern matches topic files under the memory directory.
	// It is recursive by design (aligns with Claude Code behavior).
	CandidateGlobPattern = "**/*.md"

	memoryIndexFileName = "MEMORY.md"

	defaultIndexMaxLines = 200
	defaultIndexMaxBytes = 4 * 1024

	defaultCandidateLimit       = 200
	defaultCandidatePreviewLine = 30

	defaultTopicTopK     = 5
	defaultTopicMaxLines = 200
	defaultTopicMaxBytes = 4 * 1024

	defaultMemoryWriteMaxTurns = 5

	topicSelectionToolName = "select_memories"
)

// OnError stage constants. These values are stable identifiers used to report
// best-effort failures through Config.OnError.
const (
	OnErrorStageTopicSelectionSync   = "topic_selection_sync"
	OnErrorStageTopicSelectionAsync  = "topic_selection_async"
	OnErrorStageRenderInstruction    = "render_instruction"
	OnErrorStageResolveSessionID     = "resolve_session_id"
	OnErrorStageMemoryWriteSync      = "memory_write_sync"
	OnErrorStageSnapshotMarshal      = "snapshot_marshal"
	OnErrorStageAcquireExtractionLock = "acquire_extraction_lock"
	OnErrorStageStashPendingSnapshot = "stash_pending_snapshot"
	OnErrorStageReleaseExtractionLock = "release_extraction_lock"
	OnErrorStageDecodePendingSnapshot = "decode_pending_snapshot"
	OnErrorStageMemoryWriteAsync     = "memory_write_async"
	OnErrorStageLoadPendingSnapshot  = "load_pending_snapshot"
)
