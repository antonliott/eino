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

package taskstate

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"
)

// Option configures an InMemoryManager.
type Option func(*InMemoryManager)

// WithOnEvent sets a callback that will be injected into every Handle returned by Register.
// This allows the Manager owner to receive intermediate events from all tasks without
// coupling the Manager to specific event types.
func WithOnEvent(fn func(event any)) Option {
	return func(m *InMemoryManager) {
		m.onEvent = fn
	}
}

// NewInMemoryManager creates a new in-memory Manager implementation.
func NewInMemoryManager(opts ...Option) *InMemoryManager {
	m := &InMemoryManager{
		tasks:         make(map[string]*taskRecord),
		notifications: make(chan *Notification, 256),
	}
	m.cond = sync.NewCond(&m.mu)
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// InMemoryManager is an in-memory implementation of the Manager interface.
// It is safe for concurrent use.
type InMemoryManager struct {
	mu            sync.Mutex
	cond          *sync.Cond
	tasks         map[string]*taskRecord
	seq           int64
	notifications chan *Notification
	closed        bool
	onEvent       func(event any)
}

type taskRecord struct {
	entry  Entry
	cancel context.CancelFunc
	done   bool // guards Complete/Fail idempotency
}

// Register creates a new task in StatusRunning and returns a Handle for the task owner.
// The Handle.Ctx is derived from ctx and will be cancelled when Cancel is called or the Manager is closed.
// Returns an error if the Manager has been closed.
func (m *InMemoryManager) Register(ctx context.Context, info *RegisterInfo) (*Handle, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil, fmt.Errorf("taskstate: manager is closed")
	}

	m.seq++
	id := "t_" + strconv.FormatInt(m.seq, 10)

	taskCtx, cancel := context.WithCancel(ctx)

	rec := &taskRecord{
		entry: Entry{
			ID:          id,
			Description: info.Description,
			Status:      StatusRunning,
			CreatedAt:   time.Now(),
		},
		cancel: cancel,
	}
	m.tasks[id] = rec

	h := &Handle{
		ID:  id,
		Ctx: taskCtx,
		Complete: func(result string) {
			m.complete(id, result)
		},
		Fail: func(err error) {
			m.fail(id, err)
		},
		OnEvent: m.onEvent,
	}

	return h, nil
}

// Get returns the current state of a task by ID.
// Returns (nil, false) if the task does not exist.
func (m *InMemoryManager) Get(id string) (*Entry, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.tasks[id]
	if !ok {
		return nil, false
	}
	clone := rec.entry
	return &clone, true
}

// List returns a snapshot of all tasks (both running and completed).
func (m *InMemoryManager) List() []*Entry {
	m.mu.Lock()
	defer m.mu.Unlock()

	entries := make([]*Entry, 0, len(m.tasks))
	for _, rec := range m.tasks {
		clone := rec.entry
		entries = append(entries, &clone)
	}
	return entries
}

// Cancel requests cancellation of a running task.
// The task's Handle.Ctx will be cancelled, and the task will transition to StatusCancelled.
// Returns an error if the task does not exist or is not in StatusRunning.
func (m *InMemoryManager) Cancel(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.tasks[id]
	if !ok {
		return fmt.Errorf("taskstate: task %q not found", id)
	}
	if rec.done {
		return fmt.Errorf("taskstate: task %q is not running (status: %s)", id, rec.entry.Status)
	}

	rec.done = true
	rec.cancel()
	now := time.Now()
	rec.entry.Status = StatusCancelled
	rec.entry.DoneAt = &now

	m.sendNotificationLocked(&Notification{Entry: cloneEntry(&rec.entry)})
	m.cond.Broadcast()
	return nil
}

// Notifications returns a read-only channel that receives a Notification
// each time a task reaches a terminal state.
func (m *InMemoryManager) Notifications() <-chan *Notification {
	return m.notifications
}

// HasRunning reports whether any tasks are currently in StatusRunning.
func (m *InMemoryManager) HasRunning() bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, rec := range m.tasks {
		if !rec.done {
			return true
		}
	}
	return false
}

// WaitAllDone blocks until all registered tasks have reached a terminal state,
// or until the provided context is cancelled.
func (m *InMemoryManager) WaitAllDone(ctx context.Context) error {
	// Spawn a goroutine to broadcast on ctx cancellation so cond.Wait() unblocks.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			m.cond.Broadcast()
		case <-done:
		}
	}()

	m.mu.Lock()
	defer m.mu.Unlock()

	for m.hasRunningLocked() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		m.cond.Wait()
	}
	return nil
}

// Close performs graceful shutdown of the Manager.
// It waits for all running tasks to complete (up to the ctx deadline),
// then cancels any remaining running tasks and closes the Notifications channel.
// After Close returns, Register will return an error.
func (m *InMemoryManager) Close(ctx context.Context) error {
	// Wait for running tasks to finish within the deadline.
	_ = m.WaitAllDone(ctx)

	m.mu.Lock()
	defer m.mu.Unlock()

	m.closed = true

	// Cancel any tasks still running after the wait.
	for _, rec := range m.tasks {
		if !rec.done {
			rec.done = true
			rec.cancel()
			now := time.Now()
			rec.entry.Status = StatusCancelled
			rec.entry.DoneAt = &now
			m.sendNotificationLocked(&Notification{Entry: cloneEntry(&rec.entry)})
		}
	}

	close(m.notifications)
	return nil
}

func (m *InMemoryManager) complete(id, result string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.tasks[id]
	if !ok || rec.done {
		return
	}

	rec.done = true
	now := time.Now()
	rec.entry.Status = StatusCompleted
	rec.entry.Result = result
	rec.entry.DoneAt = &now

	m.sendNotificationLocked(&Notification{Entry: cloneEntry(&rec.entry)})
	m.cond.Broadcast()
}

func (m *InMemoryManager) fail(id string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	rec, ok := m.tasks[id]
	if !ok || rec.done {
		return
	}

	rec.done = true
	rec.cancel()
	now := time.Now()
	rec.entry.Status = StatusFailed
	if err != nil {
		rec.entry.Error = err.Error()
	}
	rec.entry.DoneAt = &now

	m.sendNotificationLocked(&Notification{Entry: cloneEntry(&rec.entry)})
	m.cond.Broadcast()
}

func (m *InMemoryManager) hasRunningLocked() bool {
	for _, rec := range m.tasks {
		if !rec.done {
			return true
		}
	}
	return false
}

// sendNotificationLocked sends a notification on the buffered channel.
// Must be called with m.mu held. Uses non-blocking send to avoid deadlock.
func (m *InMemoryManager) sendNotificationLocked(n *Notification) {
	select {
	case m.notifications <- n:
	default:
	}
}

func cloneEntry(e *Entry) *Entry {
	clone := *e
	return &clone
}

// Compile-time check that InMemoryManager implements Manager.
var _ Manager = (*InMemoryManager)(nil)
