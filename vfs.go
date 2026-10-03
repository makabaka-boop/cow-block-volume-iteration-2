// Package vfs implements an in-memory virtual file volume with fixed-size
// blocks, sparse files, reference-counted block sharing (copy-on-write
// clones), a physical-block quota and optimistic concurrency control through
// a monotonically increasing volume revision.
//
// All mutating operations require the caller to present the revision it
// observed. Exactly one concurrent mutation carrying the same expected
// revision can commit; the others fail with ErrConflict without changing any
// state. Every mutation either commits completely or leaves the volume
// untouched.
package vfs

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

const int64Max = int64(^uint64(0) >> 1)

// BlockSize is the fixed physical block size in bytes.
const BlockSize = 4096

// MaxBlocks is the maximum number of physical blocks the volume may hold.
const MaxBlocks = 1024

var (
	// ErrInvalidName is returned when a file name is not a legal ASCII name.
	ErrInvalidName = errors.New("vfs: invalid file name")
	// ErrExists is returned when creating a file whose name is already used.
	ErrExists = errors.New("vfs: file already exists")
	// ErrNotFound is returned when a referenced file does not exist.
	ErrNotFound = errors.New("vfs: file not found")
	// ErrConflict is returned when a mutation expects a revision other than
	// the current volume revision.
	ErrConflict = errors.New("vfs: revision conflict")
	// ErrOutOfRange is returned when an offset is negative, an offset+length
	// sum overflows, a read starts past the end of the file, or a zero-length
	// write starts past the end of an existing file.
	ErrOutOfRange = errors.New("vfs: offset out of range")
	// ErrQuota is returned when an operation cannot reserve enough free
	// physical blocks. Nothing is mutated in that case.
	ErrQuota = errors.New("vfs: out of physical blocks")
	// ErrInvalidBatch is returned when an atomic batch is empty, has more than
	// eight steps, or contains an unknown operation.
	ErrInvalidBatch = errors.New("vfs: invalid batch")
)

// MaxBatchSteps is the maximum number of operations in one atomic batch.
const MaxBatchSteps = 8

// file is a single logical file. blocks maps a logical block index to the ID
// of a physical block. A missing key denotes a sparse hole: reading it
// yields zeros and it consumes no physical storage.
type file struct {
	name   string
	length int64
	blocks map[int64]int
}

// pblock is a physical block. refs counts how many logical block slots
// (across all files) reference it. A block with refs == 0 is free and gets
// recycled.
type pblock struct {
	data []byte
	refs int
}

// Volume is the virtual file volume. A single RWMutex guards every mutable
// structure; mutations are serialized and therefore atomic.
type Volume struct {
	mu sync.RWMutex

	blocks   map[int]*pblock // physical block ID -> block
	files    map[string]*file
	freelist []int // IDs of blocks whose reference count reached zero
	revision int64 // committed volume revision; first mutation moves 0 -> 1
}

// FileInfo describes a single file. PhysicalBytes counts referenced slots;
// a block shared by a clone is counted per referencing file here and once in
// the volume totals.
type FileInfo struct {
	Name           string `json:"name"`
	Length         int64  `json:"length"`
	PhysicalBlocks int    `json:"physicalBlocks"`
	PhysicalBytes  int64  `json:"physicalBytes"`
}

// Stats describes volume occupancy. UsedBlocks is the number of distinct
// physical blocks currently allocated; shared blocks are counted once.
type Stats struct {
	Revision     int64      `json:"revision"`
	LogicalFiles int        `json:"logicalFiles"`
	LogicalBytes int64      `json:"logicalBytes"`
	SparseBytes  int64      `json:"sparseBytes"`
	UsedBlocks   int        `json:"usedBlocks"`
	UsedBytes    int64      `json:"usedBytes"`
	FreeBlocks   int        `json:"freeBlocks"`
	MaxBlocks    int        `json:"maxBlocks"`
	BlockSize    int        `json:"blockSize"`
	Files        []FileInfo `json:"files"`
}

// BatchStep is one operation in an atomic Batch. Op is one of "create",
// "clone", "write", "truncate", or "delete". Field names follow the existing
// single-operation parameters: Name/Src/Dst, Offset, Length and Data.
type BatchStep struct {
	Op     string `json:"op"`
	Name   string `json:"name,omitempty"`
	Src    string `json:"src,omitempty"`
	Dst    string `json:"dst,omitempty"`
	Offset int64  `json:"offset,omitempty"`
	Length int64  `json:"length,omitempty"`
	Data   []byte `json:"data,omitempty"`
}

// BatchError identifies a failed batch step. Step is its one-based position in
// the request.
type BatchError struct {
	Step int
	Err  error
}

func (e *BatchError) Error() string {
	return fmt.Sprintf("vfs: batch step %d: %v", e.Step, e.Err)
}

func (e *BatchError) Unwrap() error { return e.Err }

// New returns an empty volume at revision 0.
func New() *Volume {
	return &Volume{
		blocks: make(map[int]*pblock),
		files:  make(map[string]*file),
	}
}

// snapshotLocked returns an independent deep copy of the committed volume.
// Batch preparation mutates the copy; on failure it is simply discarded, so
// readers and the real volume keep the pre-batch state.
func (v *Volume) snapshotLocked() *Volume {
	cp := &Volume{
		blocks:   make(map[int]*pblock, len(v.blocks)),
		files:    make(map[string]*file, len(v.files)),
		freelist: append([]int(nil), v.freelist...),
		revision: v.revision,
	}
	for id, b := range v.blocks {
		data := append([]byte(nil), b.data...)
		cp.blocks[id] = &pblock{data: data, refs: b.refs}
	}
	for name, f := range v.files {
		nf := &file{name: f.name, length: f.length, blocks: make(map[int64]int, len(f.blocks))}
		for idx, id := range f.blocks {
			nf.blocks[idx] = id
		}
		cp.files[name] = nf
	}
	return cp
}

// ValidateName reports whether name is a legal ASCII file name. Names are
// 1..255 bytes, must start with an ASCII letter or digit, and may otherwise
// contain letters, digits, dot, underscore and hyphen.
func ValidateName(name string) bool {
	n := len(name)
	if n == 0 || n > 255 {
		return false
	}
	for i := 0; i < n; i++ {
		c := name[i]
		isAlnum := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
		if i == 0 {
			if !isAlnum {
				return false
			}
			continue
		}
		if !isAlnum && c != '.' && c != '_' && c != '-' {
			return false
		}
	}
	return true
}

// Revision returns the current volume revision.
func (v *Volume) Revision() int64 {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.revision
}

// allocLocked hands out one free physical block ID, reusing blocks whose
// reference count previously dropped to zero.
func (v *Volume) allocLocked() int {
	if n := len(v.freelist); n > 0 {
		id := v.freelist[n-1]
		v.freelist = v.freelist[:n-1]
		return id
	}
	return len(v.blocks)
}

// decLocked drops one reference from a physical block and recycles it once
// no logical slot points to it anymore.
func (v *Volume) decLocked(id int) {
	b := v.blocks[id]
	b.refs--
	if b.refs == 0 {
		delete(v.blocks, id)
		v.freelist = append(v.freelist, id)
	}
}

// cowLocked ensures the block backing logical slot idx is privately writable,
// copying it on write when it is shared. A sparse hole is backed by a fresh
// zero-filled block.
func (v *Volume) cowLocked(f *file, idx int64) {
	id, ok := f.blocks[idx]
	if !ok {
		nid := v.allocLocked()
		v.blocks[nid] = &pblock{data: make([]byte, BlockSize), refs: 1}
		f.blocks[idx] = nid
		return
	}
	if v.blocks[id].refs == 1 {
		return // exclusively owned; safe to overwrite in place
	}
	nid := v.allocLocked()
	nb := make([]byte, BlockSize)
	copy(nb, v.blocks[id].data)
	v.blocks[nid] = &pblock{data: nb, refs: 1}
	f.blocks[idx] = nid
	v.decLocked(id)
}

// createLocked performs Create after revision checking. It does not advance
// the revision; callers own the single revision increment for their commit.
func (v *Volume) createLocked(name string) error {
	if _, ok := v.files[name]; ok {
		return ErrExists
	}
	v.files[name] = &file{name: name, blocks: make(map[int64]int)}
	return nil
}

// Create creates an empty file.
func (v *Volume) Create(name string, expected int64) (newRev int64, err error) {
	if !ValidateName(name) {
		return 0, ErrInvalidName
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}
	if err := v.createLocked(name); err != nil {
		return v.revision, err
	}
	v.revision++
	return v.revision, nil
}

// cloneLocked performs Clone after revision checking. It does not advance the
// revision; callers own the single revision increment for their commit.
func (v *Volume) cloneLocked(src, dst string) error {
	s, ok := v.files[src]
	if !ok {
		return ErrNotFound
	}
	if _, ok := v.files[dst]; ok {
		return ErrExists
	}
	nf := &file{name: dst, length: s.length, blocks: make(map[int64]int, len(s.blocks))}
	for idx, id := range s.blocks {
		nf.blocks[idx] = id
		v.blocks[id].refs++
	}
	v.files[dst] = nf
	return nil
}

// Clone creates dst as an instant copy of src. The logical block table is
// duplicated; physical blocks are shared (reference counts bumped) and only
// physically copied on the first write that touches them.
func (v *Volume) Clone(src, dst string, expected int64) (newRev int64, err error) {
	if !ValidateName(src) || !ValidateName(dst) {
		return 0, ErrInvalidName
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}
	if err := v.cloneLocked(src, dst); err != nil {
		return v.revision, err
	}
	v.revision++
	return v.revision, nil
}

func (v *Volume) writeLocked(name string, offset int64, data []byte) (bool, error) {
	f, ok := v.files[name]
	if !ok {
		return false, ErrNotFound
	}
	if len(data) == 0 {
		if offset > f.length {
			return false, ErrOutOfRange
		}
		return false, nil
	}

	end := offset + int64(len(data))

	// Reservation phase: count every block we must allocate (holes and
	// shared blocks that need copy-on-write).
	need := 0
	first := offset / BlockSize
	last := (end - 1) / BlockSize
	for idx := first; idx <= last; idx++ {
		id, present := f.blocks[idx]
		if !present || v.blocks[id].refs > 1 {
			need++
		}
	}
	if len(v.blocks)+need > MaxBlocks {
		return false, ErrQuota
	}

	// Commit phase: every remaining step is infallible.
	d := data
	off := offset
	for len(d) > 0 {
		idx := off / BlockSize
		v.cowLocked(f, idx)
		start := off - idx*BlockSize
		n := copy(v.blocks[f.blocks[idx]].data[start:], d)
		d = d[n:]
		off += int64(n)
	}
	if end > f.length {
		f.length = end
	}
	return true, nil
}

// Write overwrites len(data) bytes at offset, growing the file and creating
// sparse holes as needed. Shared blocks are copied before the first
// modification. It fails atomically (ErrQuota or ErrOutOfRange): no partial
// writes and no leaked or miscounted references.
//
// A zero-length write is a no-op (including its revision) when offset is at or
// before EOF; it cannot extend a file. A zero-length write starting past EOF
// returns ErrOutOfRange without changing state.
func (v *Volume) Write(name string, offset int64, data []byte, expected int64) (newRev int64, err error) {
	if !ValidateName(name) {
		return 0, ErrInvalidName
	}
	if offset < 0 {
		return 0, ErrOutOfRange
	}
	if len(data) > 0 && offset > int64Max-int64(len(data)) {
		return 0, ErrOutOfRange
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}
	mutated, err := v.writeLocked(name, offset, data)
	if err != nil {
		return v.revision, err
	}
	if mutated {
		v.revision++
	}
	return v.revision, nil
}

// Truncate sets the logical length. Shrinking drops blocks fully past the new
// end and zeros the tail of the block straddling the boundary (copying it
// first if shared, so clones keep their data). Growing only extends the
// logical size; the new region is a sparse hole. A quota failure while
// shrinking is atomic: nothing is removed or copied.
func (v *Volume) Truncate(name string, length int64, expected int64) (newRev int64, err error) {
	if !ValidateName(name) {
		return 0, ErrInvalidName
	}
	if length < 0 {
		return 0, ErrOutOfRange
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}
	if err := v.truncateLocked(name, length); err != nil {
		return v.revision, err
	}
	v.revision++
	return v.revision, nil
}

// truncateLocked performs Truncate after revision checking. It does not
// advance the revision; callers own the single revision increment.
func (v *Volume) truncateLocked(name string, length int64) error {
	f, ok := v.files[name]
	if !ok {
		return ErrNotFound
	}

	if length >= f.length {
		// Growth: no physical work. Bytes between the old partial tail and
		// any future write remain logical zeros / holes.
		f.length = length
		return nil
	}

	// Shrink. Classify mapped slots without mutating anything.
	lastKeep := int64(-1)
	if length > 0 {
		lastKeep = (length - 1) / BlockSize
	}
	var straddle int64
	hasStraddle := length%BlockSize != 0
	if hasStraddle {
		straddle = lastKeep
	}

	var dropped []int // logical slots removed entirely
	for idx := range f.blocks {
		if idx > lastKeep {
			dropped = append(dropped, int(idx))
		}
	}

	// Physical blocks that become unreferenced after dropping those slots.
	reclaim := 0
	for _, idx := range dropped {
		if v.blocks[f.blocks[int64(idx)]].refs == 1 {
			reclaim++
		}
	}
	// The straddling block needs a private copy only when it exists, is
	// shared and its tail must be zeroed.
	straddleNeedsAlloc := false
	if hasStraddle {
		if id, present := f.blocks[straddle]; present && v.blocks[id].refs > 1 {
			straddleNeedsAlloc = true
		}
	}
	if straddleNeedsAlloc && len(v.blocks)-reclaim+1 > MaxBlocks {
		return ErrQuota
	}

	// Commit phase.
	for _, idx := range dropped {
		id := f.blocks[int64(idx)]
		delete(f.blocks, int64(idx))
		v.decLocked(id)
	}
	if hasStraddle {
		// cowLocked is a no-op for an exclusively owned block and, crucially,
		// does not materialize a sparse hole unless it is about to be zeroed
		// below; a hole stays a hole.
		if id, present := f.blocks[straddle]; present {
			if v.blocks[id].refs > 1 {
				v.cowLocked(f, straddle)
			}
			id = f.blocks[straddle]
			for i := length % BlockSize; i < BlockSize; i++ {
				v.blocks[id].data[i] = 0
			}
		}
	}
	f.length = length
	return nil
}

func (v *Volume) deleteLocked(name string) error {
	f, ok := v.files[name]
	if !ok {
		return ErrNotFound
	}
	for _, id := range f.blocks {
		v.decLocked(id)
	}
	delete(v.files, name)
	return nil
}

// Delete removes a file and drops all its block references; physical blocks
// shared with other files survive.
func (v *Volume) Delete(name string, expected int64) (newRev int64, err error) {
	if !ValidateName(name) {
		return 0, ErrInvalidName
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}
	if err := v.deleteLocked(name); err != nil {
		return v.revision, err
	}
	v.revision++
	return v.revision, nil
}

// validateBatchStep checks request-independent and revision-independent
// parameter validity using the same rules as the corresponding single-step
// operation.
func validateBatchStep(s BatchStep) error {
	switch s.Op {
	case "create":
		if !ValidateName(s.Name) {
			return ErrInvalidName
		}
	case "clone":
		if !ValidateName(s.Src) || !ValidateName(s.Dst) {
			return ErrInvalidName
		}
	case "write":
		if !ValidateName(s.Name) {
			return ErrInvalidName
		}
		if s.Offset < 0 {
			return ErrOutOfRange
		}
		if len(s.Data) > 0 && s.Offset > int64Max-int64(len(s.Data)) {
			return ErrOutOfRange
		}
	case "truncate":
		if !ValidateName(s.Name) {
			return ErrInvalidName
		}
		if s.Length < 0 {
			return ErrOutOfRange
		}
	case "delete":
		if !ValidateName(s.Name) {
			return ErrInvalidName
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}

// Batch applies up to eight operations in order as one transaction. The steps
// operate on a private staging copy: later steps observe earlier created files,
// deletions and writes, and each quota/copy-on-write decision sees that staged
// state. On success the staged maps, bytes, reference counts, freelist and
// statistics are published together and the revision advances exactly once. If
// any step fails, its one-based index and cause are returned and the committed
// volume is unchanged.
func (v *Volume) Batch(steps []BatchStep, expected int64) (newRev int64, err error) {
	if len(steps) == 0 || len(steps) > MaxBatchSteps {
		return 0, ErrInvalidBatch
	}
	for i, step := range steps {
		if err := validateBatchStep(step); err != nil {
			return 0, &BatchError{Step: i + 1, Err: err}
		}
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if expected != v.revision {
		return v.revision, ErrConflict
	}

	stage := v.snapshotLocked()
	for i, step := range steps {
		var applyErr error
		switch step.Op {
		case "create":
			applyErr = stage.createLocked(step.Name)
		case "clone":
			applyErr = stage.cloneLocked(step.Src, step.Dst)
		case "write":
			_, applyErr = stage.writeLocked(step.Name, step.Offset, step.Data)
		case "truncate":
			applyErr = stage.truncateLocked(step.Name, step.Length)
		case "delete":
			applyErr = stage.deleteLocked(step.Name)
		}
		if applyErr != nil {
			return v.revision, &BatchError{Step: i + 1, Err: applyErr}
		}
	}

	v.blocks = stage.blocks
	v.files = stage.files
	v.freelist = stage.freelist
	v.revision++
	return v.revision, nil
}

// Read returns up to length bytes starting at offset. A negative length reads
// to the end of the file. Sparse holes yield zero bytes. Starting a read
// exactly at end-of-file returns an empty slice; starting past it is
// ErrOutOfRange. A positive length whose sum with offset overflows int64 also
// returns ErrOutOfRange.
func (v *Volume) Read(name string, offset, length int64) ([]byte, error) {
	data, _, err := v.ReadWithRevision(name, offset, length)
	return data, err
}

// ReadWithRevision is Read and also returns the revision of the file snapshot
// represented by the returned bytes.
func (v *Volume) ReadWithRevision(name string, offset, length int64) (data []byte, revision int64, err error) {
	if !ValidateName(name) {
		return nil, 0, ErrInvalidName
	}
	if offset < 0 {
		return nil, 0, ErrOutOfRange
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	f, ok := v.files[name]
	if !ok {
		return nil, v.revision, ErrNotFound
	}
	if offset > f.length {
		return nil, v.revision, ErrOutOfRange
	}
	if length >= 0 && length > int64Max-offset {
		return nil, v.revision, ErrOutOfRange
	}
	if length < 0 || offset+length > f.length {
		length = f.length - offset
	}
	buf := make([]byte, length)
	if length == 0 {
		return buf, v.revision, nil
	}
	first := offset / BlockSize
	last := (offset + length - 1) / BlockSize
	for idx := first; idx <= last; idx++ {
		id, present := f.blocks[idx]
		if !present {
			continue // hole stays zero
		}
		blkStart := idx * BlockSize
		lo := offset - blkStart
		if lo < 0 {
			lo = 0
		}
		hi := int64(BlockSize)
		if blkStart+BlockSize > offset+length {
			hi = offset + length - blkStart
		}
		copy(buf[blkStart+lo-offset:], v.blocks[id].data[lo:hi])
	}
	return buf, v.revision, nil
}

// Stats returns logical length and physical occupancy for the volume and each
// file. UsedBlocks counts distinct physical blocks; a shared block is counted
// once. SparseBytes is the logical-file space not backed by a physical block
// slot; a materialized partial last block only counts its in-file bytes as
// backed. PhysicalBytes continues to report whole allocated blocks.
func (v *Volume) Stats() Stats {
	v.mu.RLock()
	defer v.mu.RUnlock()
	st := Stats{
		Revision:     v.revision,
		LogicalFiles: len(v.files),
		UsedBlocks:   len(v.blocks),
		UsedBytes:    int64(len(v.blocks)) * BlockSize,
		FreeBlocks:   MaxBlocks - len(v.blocks),
		MaxBlocks:    MaxBlocks,
		BlockSize:    BlockSize,
		Files:        make([]FileInfo, 0, len(v.files)),
	}
	names := make([]string, 0, len(v.files))
	for n := range v.files {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		f := v.files[n]
		phys := len(f.blocks)
		st.LogicalBytes += f.length
		var backed int64
		for idx := range f.blocks {
			start := idx * BlockSize
			if start >= f.length {
				continue
			}
			end := start + BlockSize
			if end > f.length {
				end = f.length
			}
			backed += end - start
		}
		st.SparseBytes += f.length - backed
		st.Files = append(st.Files, FileInfo{
			Name:           n,
			Length:         f.length,
			PhysicalBlocks: phys,
			PhysicalBytes:  int64(phys) * BlockSize,
		})
	}
	return st
}

// DebugBlockID returns the physical block ID backing a file's logical slot,
// or -1 for a hole / missing file. It is intended for tests.
func (v *Volume) DebugBlockID(name string, idx int64) int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	f, ok := v.files[name]
	if !ok {
		return -1
	}
	id, present := f.blocks[idx]
	if !present {
		return -1
	}
	return id
}
