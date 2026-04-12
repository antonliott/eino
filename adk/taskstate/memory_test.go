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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// closeWithTimeout closes the manager with a short timeout to avoid blocking on uncompleted tasks.
func closeWithTimeout(mgr *InMemoryManager) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_ = mgr.Close(ctx)
}

func TestRegisterAndGet(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "test task"})
	require.NoError(t, err)
	assert.NotEmpty(t, h.ID)
	assert.NotNil(t, h.Ctx)
	assert.NotNil(t, h.Complete)
	assert.NotNil(t, h.Fail)

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, h.ID, entry.ID)
	assert.Equal(t, "test task", entry.Description)
	assert.Equal(t, StatusRunning, entry.Status)
	assert.Nil(t, entry.DoneAt)
}

func TestGetNotFound(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	entry, ok := mgr.Get("nonexistent")
	assert.False(t, ok)
	assert.Nil(t, entry)
}

func TestComplete(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Complete("done")

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, StatusCompleted, entry.Status)
	assert.Equal(t, "done", entry.Result)
	assert.NotNil(t, entry.DoneAt)
}

func TestFail(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Fail(fmt.Errorf("something went wrong"))

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, StatusFailed, entry.Status)
	assert.Equal(t, "something went wrong", entry.Error)
	assert.NotNil(t, entry.DoneAt)
}

func TestCancel(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	err = mgr.Cancel(h.ID)
	require.NoError(t, err)

	// Context should be cancelled.
	select {
	case <-h.Ctx.Done():
	default:
		t.Fatal("expected context to be cancelled")
	}

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, StatusCancelled, entry.Status)
	assert.NotNil(t, entry.DoneAt)
}

func TestCancelNotFound(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	err := mgr.Cancel("nonexistent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestCancelAlreadyDone(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Complete("result")

	err = mgr.Cancel(h.ID)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not running")
}

func TestIdempotentComplete(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Complete("first")
	h.Complete("second") // should be no-op

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, "first", entry.Result)
}

func TestIdempotentFail(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Fail(fmt.Errorf("first"))
	h.Fail(fmt.Errorf("second")) // should be no-op

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, "first", entry.Error)
}

func TestCompleteAfterFail(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)

	h.Fail(fmt.Errorf("error"))
	h.Complete("result") // should be no-op

	entry, ok := mgr.Get(h.ID)
	require.True(t, ok)
	assert.Equal(t, StatusFailed, entry.Status)
}

func TestNotifications(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h1, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task1"})
	require.NoError(t, err)
	h2, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task2"})
	require.NoError(t, err)

	h1.Complete("result1")
	h2.Fail(fmt.Errorf("error2"))

	// Collect notifications.
	var notifications []*Notification
	timeout := time.After(time.Second)
	for i := 0; i < 2; i++ {
		select {
		case n := <-mgr.Notifications():
			notifications = append(notifications, n)
		case <-timeout:
			t.Fatal("timed out waiting for notifications")
		}
	}

	assert.Len(t, notifications, 2)
	ids := map[string]bool{
		notifications[0].Entry.ID: true,
		notifications[1].Entry.ID: true,
	}
	assert.True(t, ids[h1.ID])
	assert.True(t, ids[h2.ID])
}

func TestList(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h1, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "a"})
	h2, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "b"})
	h1.Complete("done")

	entries := mgr.List()
	assert.Len(t, entries, 2)

	byID := make(map[string]*Entry)
	for _, e := range entries {
		byID[e.ID] = e
	}
	assert.Equal(t, StatusCompleted, byID[h1.ID].Status)
	assert.Equal(t, StatusRunning, byID[h2.ID].Status)
}

func TestHasRunning(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	assert.False(t, mgr.HasRunning())

	h, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	assert.True(t, mgr.HasRunning())

	h.Complete("done")
	assert.False(t, mgr.HasRunning())
}

func TestWaitAllDone(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h1, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "task1"})
	h2, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "task2"})

	go func() {
		time.Sleep(50 * time.Millisecond)
		h1.Complete("r1")
		time.Sleep(50 * time.Millisecond)
		h2.Complete("r2")
	}()

	err := mgr.WaitAllDone(context.Background())
	assert.NoError(t, err)
	assert.False(t, mgr.HasRunning())
}

func TestWaitAllDoneTimeout(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	_, _ = mgr.Register(context.Background(), &RegisterInfo{Description: "task"})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := mgr.WaitAllDone(ctx)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestWaitAllDoneNoTasks(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	err := mgr.WaitAllDone(context.Background())
	assert.NoError(t, err)
}

func TestClose(t *testing.T) {
	mgr := NewInMemoryManager()

	h, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := mgr.Close(ctx)
	assert.NoError(t, err)

	// Task should be cancelled.
	select {
	case <-h.Ctx.Done():
	default:
		t.Fatal("expected context to be cancelled after Close")
	}

	// Notifications channel should be closed.
	_, ok := <-mgr.Notifications()
	// Either get remaining notification or channel closed
	if ok {
		// Drain remaining
		for range mgr.Notifications() {
		}
	}

	// Register after close should fail.
	_, err = mgr.Register(context.Background(), &RegisterInfo{Description: "new"})
	assert.Error(t, err)
}

func TestRegisterAfterClose(t *testing.T) {
	mgr := NewInMemoryManager()
	_ = mgr.Close(context.Background())

	_, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}

func TestWithOnEvent(t *testing.T) {
	var mu sync.Mutex
	var events []any

	mgr := NewInMemoryManager(WithOnEvent(func(event any) {
		mu.Lock()
		events = append(events, event)
		mu.Unlock()
	}))
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)
	require.NotNil(t, h.OnEvent)

	h.OnEvent("event1")
	h.OnEvent("event2")

	mu.Lock()
	assert.Equal(t, []any{"event1", "event2"}, events)
	mu.Unlock()
}

func TestWithOnEventNil(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	require.NoError(t, err)
	assert.Nil(t, h.OnEvent)
}

func TestConcurrentRegisterComplete(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	const n = 50
	var wg sync.WaitGroup
	wg.Add(n)

	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			h, err := mgr.Register(context.Background(), &RegisterInfo{
				Description: fmt.Sprintf("task-%d", i),
			})
			require.NoError(t, err)
			h.Complete(fmt.Sprintf("result-%d", i))
		}(i)
	}

	wg.Wait()
	assert.False(t, mgr.HasRunning())
	assert.Len(t, mgr.List(), n)
}

func TestUniqueIDs(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	ids := make(map[string]bool)
	for i := 0; i < 100; i++ {
		h, err := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
		require.NoError(t, err)
		assert.False(t, ids[h.ID], "duplicate ID: %s", h.ID)
		ids[h.ID] = true
	}
}

func TestCancelNotification(t *testing.T) {
	mgr := NewInMemoryManager()
	defer closeWithTimeout(mgr)

	h, _ := mgr.Register(context.Background(), &RegisterInfo{Description: "task"})
	_ = mgr.Cancel(h.ID)

	select {
	case n := <-mgr.Notifications():
		assert.Equal(t, StatusCancelled, n.Entry.Status)
		assert.Equal(t, h.ID, n.Entry.ID)
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for cancel notification")
	}
}
