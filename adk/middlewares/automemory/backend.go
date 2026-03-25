package automemory

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/cloudwego/eino/adk/filesystem"
)

type Backend interface {
	// Read reads file content with support for line-based offset and limit.
	//
	// Returns:
	//   - string: The file content read
	//   - error: Error if file does not exist or read fails
	Read(ctx context.Context, req *ReadRequest) (*FileContent, error)

	// GlobInfo returns file information matching the glob pattern.
	//
	// Returns:
	//   - []FileInfo: List of matching file information
	//   - error: Error if the pattern is invalid or operation fails
	GlobInfo(ctx context.Context, req *GlobInfoRequest) ([]FileInfo, error)

	// Write writes file content.
	Write(ctx context.Context, req *WriteRequest) error

	// Edit performs a string replacement edit on a file.
	Edit(ctx context.Context, req *EditRequest) error
}

type ReadRequest = filesystem.ReadRequest
type FileContent = filesystem.FileContent
type GlobInfoRequest = filesystem.GlobInfoRequest
type FileInfo = filesystem.FileInfo
type WriteRequest = filesystem.WriteRequest
type EditRequest = filesystem.EditRequest

type fsBackend struct {
	backend   Backend
	baseDir   string
	baseClean string
}

func newFSBackend(backend Backend, baseDir string) (*fsBackend, error) {
	if backend == nil {
		return nil, fmt.Errorf("fs backend: nil backend")
	}
	if baseDir == "" {
		return nil, fmt.Errorf("fs backend: empty base dir")
	}
	baseClean := filepath.Clean(baseDir)
	return &fsBackend{
		backend:   backend,
		baseDir:   baseDir,
		baseClean: baseClean,
	}, nil
}

func (f *fsBackend) resolveFilePath(p string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("fs backend: empty path")
	}
	// The filesystem tools may pass relative paths. Interpret them as relative to the
	// configured base directory and prevent path traversal.
	if !filepath.IsAbs(p) {
		p = filepath.Join(f.baseDir, p)
	}
	p = filepath.Clean(p)
	if p != f.baseClean && !strings.HasPrefix(p, f.baseClean+string(filepath.Separator)) {
		return "", fmt.Errorf("fs backend: path out of bounds: %s", p)
	}
	return p, nil
}

func (f *fsBackend) resolveDirPath(p string) (string, error) {
	if p == "" {
		return f.baseClean, nil
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(f.baseDir, p)
	}
	p = filepath.Clean(p)
	if p != f.baseClean && !strings.HasPrefix(p, f.baseClean+string(filepath.Separator)) {
		return "", fmt.Errorf("fs backend: dir out of bounds: %s", p)
	}
	return p, nil
}

func (f *fsBackend) Read(ctx context.Context, req *ReadRequest) (*FileContent, error) {
	if req == nil {
		return nil, fmt.Errorf("read: invalid request")
	}
	fp, err := f.resolveFilePath(req.FilePath)
	if err != nil {
		return nil, err
	}
	n := *req
	n.FilePath = fp
	return f.backend.Read(ctx, &n)
}

func (f *fsBackend) Write(ctx context.Context, req *WriteRequest) error {
	if req == nil {
		return fmt.Errorf("write: invalid request")
	}
	fp, err := f.resolveFilePath(req.FilePath)
	if err != nil {
		return err
	}
	n := *req
	n.FilePath = fp
	return f.backend.Write(ctx, &n)
}

func (f *fsBackend) Edit(ctx context.Context, req *EditRequest) error {
	if req == nil {
		return fmt.Errorf("edit: invalid request")
	}
	fp, err := f.resolveFilePath(req.FilePath)
	if err != nil {
		return err
	}
	n := *req
	n.FilePath = fp
	return f.backend.Edit(ctx, &n)
}

func (f *fsBackend) GlobInfo(ctx context.Context, req *GlobInfoRequest) ([]FileInfo, error) {
	if req == nil {
		return nil, fmt.Errorf("glob: invalid request")
	}
	if req.Pattern == "" {
		return nil, fmt.Errorf("glob: invalid request")
	}

	pathAbs, err := f.resolveDirPath(req.Path)
	if err != nil {
		return nil, err
	}

	pattern := req.Pattern
	// If an absolute pattern is provided, rewrite it into a relative pattern under
	// an in-bounds base directory to match the semantics of our backends.
	if filepath.IsAbs(pattern) {
		cp := filepath.Clean(pattern)
		// Prefer resolving relative to the requested search path if possible.
		if cp == pathAbs {
			pattern = "."
		} else if strings.HasPrefix(cp, pathAbs+string(filepath.Separator)) {
			rel, rerr := filepath.Rel(pathAbs, cp)
			if rerr != nil {
				return nil, rerr
			}
			pattern = filepath.ToSlash(rel)
		} else if strings.HasPrefix(cp, f.baseClean+string(filepath.Separator)) {
			rel, rerr := filepath.Rel(f.baseClean, cp)
			if rerr != nil {
				return nil, rerr
			}
			pattern = filepath.ToSlash(rel)
			pathAbs = f.baseClean
		} else {
			return nil, fmt.Errorf("fs backend: glob pattern out of bounds: %s", cp)
		}
	} else {
		pattern = filepath.ToSlash(pattern)
	}

	n := *req
	n.Path = pathAbs
	n.Pattern = pattern
	return f.backend.GlobInfo(ctx, &n)
}

func (f *fsBackend) LsInfo(_ context.Context, _ *filesystem.LsInfoRequest) ([]filesystem.FileInfo, error) {
	// Explicitly disabled in automemory's filesystem middleware config.
	return nil, fmt.Errorf("ls: disabled")
}

func (f *fsBackend) GrepRaw(_ context.Context, _ *filesystem.GrepRequest) ([]filesystem.GrepMatch, error) {
	// Explicitly disabled in automemory's filesystem middleware config.
	return nil, fmt.Errorf("grep: disabled")
}
