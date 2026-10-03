package vfs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// ---------------------------------------------------------------------------
// Naive-model batch oracle.
//
// The model stages a batch on a deep copy of itself — full byte arrays plus
// the independent refcount table — exactly like Volume.Batch stages on a
// snapshot of the volume. Commit swaps the staged state in and bumps the
// revision once; a failure returns the failing step's index and leaves the
// model untouched.
// ---------------------------------------------------------------------------

// modelStaticKind mirrors Step.validateStatic: everything checkable without
// volume state, in the same order.
func modelStaticKind(s Step) errKind {
	switch s.Op {
	case OpCreate, OpDelete:
		if !ValidateName(s.Name) {
			return kBadName
		}
	case OpClone:
		if !ValidateName(s.Src) || !ValidateName(s.Dst) {
			return kBadName
		}
	case OpWrite:
		if !ValidateName(s.Name) {
			return kBadName
		}
		if s.Offset < 0 || (len(s.Data) > 0 && s.Offset > int64Max-int64(len(s.Data))) {
			return kRange
		}
	case OpTruncate:
		if !ValidateName(s.Name) {
			return kBadName
		}
		if s.Length < 0 {
			return kRange
		}
	default:
		return kBadBatch
	}
	return kOK
}

func (m *model) snapshot() *model {
	s := &model{
		files:    make(map[string]*modelFile, len(m.files)),
		blocks:   make(map[int]*modelBlock, len(m.blocks)),
		revision: m.revision,
		nextID:   m.nextID,
	}
	for name, f := range m.files {
		nf := &modelFile{data: append([]byte(nil), f.data...), phys: make(map[int64]int, len(f.phys))}
		for idx, id := range f.phys {
			nf.phys[idx] = id
		}
		s.files[name] = nf
	}
	for id, b := range m.blocks {
		s.blocks[id] = &modelBlock{id: b.id, refs: b.refs}
	}
	return s
}

func (m *model) apply(s Step) errKind {
	switch s.Op {
	case OpCreate:
		return m.createStaged(s.Name)
	case OpClone:
		return m.cloneStaged(s.Src, s.Dst)
	case OpWrite:
		k, _ := m.writeStaged(s.Name, s.Offset, s.Data)
		return k
	case OpTruncate:
		return m.truncateStaged(s.Name, s.Length)
	case OpDelete:
		return m.delStaged(s.Name)
	default:
		return kBadBatch
	}
}

// batch mirrors Volume.Batch: static validation of every step up front, one
// revision check, staged execution in order, all-or-nothing commit with a
// single revision bump.
func (m *model) batch(steps []Step, rev int64) (errKind, int) {
	if len(steps) == 0 || len(steps) > MaxBatchSteps {
		return kBadBatch, -1
	}
	for i, s := range steps {
		if k := modelStaticKind(s); k != kOK {
			return k, i
		}
	}
	if rev != m.revision {
		return kConflict, -1
	}
	st := m.snapshot()
	for i, s := range steps {
		if k := st.apply(s); k != kOK {
			return k, i
		}
	}
	m.files = st.files
	m.blocks = st.blocks
	m.nextID = st.nextID
	m.revision++
	return kOK, -1
}

// ---------------------------------------------------------------------------
// HTTP client helpers.
// ---------------------------------------------------------------------------

type batchResp struct {
	Revision int64  `json:"revision"`
	Error    string `json:"error"`
	Step     int    `json:"step"`
}

func decodeBatchResp(resp *http.Response) batchResp {
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var br batchResp
	_ = json.Unmarshal(raw, &br)
	return br
}

func (c *apiClient) batch(steps []Step, rev int64) (int, batchResp) {
	body, err := json.Marshal(map[string]any{"revision": rev, "steps": steps})
	if err != nil {
		panic(err)
	}
	resp, err := c.hc.Post(c.srv.URL+"/batch", "application/json", bytes.NewReader(body))
	if err != nil {
		panic(err)
	}
	return resp.StatusCode, decodeBatchResp(resp)
}

func (c *apiClient) statsT(t *testing.T) Stats {
	t.Helper()
	resp, err := c.hc.Get(c.srv.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var st Stats
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	return st
}

// kindToStatus maps a model outcome to the exact HTTP status the server must
// produce. Exists vs revision conflict need no ambiguity bucket here: the
// failing-step index distinguishes them (conflict is always step -1).
func kindToStatus(k errKind) int {
	switch k {
	case kBadName, kBadBatch:
		return http.StatusBadRequest
	case kExists, kConflict:
		return http.StatusConflict
	case kNotFound:
		return http.StatusNotFound
	case kRange:
		return http.StatusRequestedRangeNotSatisfiable
	case kQuota:
		return http.StatusInsufficientStorage
	default:
		return http.StatusInternalServerError
	}
}

// checkBatch compares the server's batch response with the model oracle:
// same outcome, same failing step, and (on success) the same new revision.
func checkBatch(t *testing.T, ctx string, status int, br batchResp, mk errKind, mstep int, m *model) {
	t.Helper()
	if mk == kOK {
		if status != http.StatusOK {
			t.Fatalf("[%s] batch status=%d want 200 (err=%q step=%d)", ctx, status, br.Error, br.Step)
		}
		if br.Revision != m.revision {
			t.Fatalf("[%s] batch revision=%d model=%d", ctx, br.Revision, m.revision)
		}
		return
	}
	if want := kindToStatus(mk); status != want {
		t.Fatalf("[%s] batch status=%d want %d (model=%v err=%q)", ctx, status, want, mk, br.Error)
	}
	if br.Step != mstep {
		t.Fatalf("[%s] batch failing step=%d model=%d", ctx, br.Step, mstep)
	}
}

func mustOK(t *testing.T, desc string, mk errKind, status int) {
	t.Helper()
	if mk != kOK || status != http.StatusOK {
		t.Fatalf("%s: model=%v status=%d", desc, mk, status)
	}
}

// ---------------------------------------------------------------------------
// Deterministic scenarios.
// ---------------------------------------------------------------------------

// TestBatchChainForkHTTP runs a full eight-step chain in one batch: a clone,
// copy-on-write forks on both sides of the shared blocks, files created and
// cloned inside the batch, a truncate and a re-grow. The revision must
// advance exactly once and the response must match the final statistics.
func TestBatchChainForkHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	rev := int64(0)
	nr, st := c.create("src", rev)
	mustOK(t, "create src", m.create("src", rev), st)
	rev = nr
	nr, st = c.write("src", 0, bytesFill(2*BlockSize, 0x11), rev)
	mustOK(t, "write src", m.write("src", 0, bytesFill(2*BlockSize, 0x11), rev), st)
	rev = nr

	steps := []Step{
		{Op: OpClone, Src: "src", Dst: "dst"},                                   // shares both blocks
		{Op: OpWrite, Name: "dst", Offset: 0, Data: []byte("FORK")},             // forks dst block 0
		{Op: OpWrite, Name: "src", Offset: BlockSize, Data: bytesFill(8, 0x22)}, // forks src block 1
		{Op: OpCreate, Name: "meta"},                                            // created mid-batch...
		{Op: OpWrite, Name: "meta", Offset: 0, Data: []byte("m1")},              // ...written...
		{Op: OpClone, Src: "meta", Dst: "meta2"},                                // ...and cloned by later steps
		{Op: OpTruncate, Name: "dst", Length: BlockSize},                        // drops dst's shared block 1
		{Op: OpWrite, Name: "dst", Offset: BlockSize, Data: bytesFill(8, 0x33)}, // fresh block 1
	}
	status, br := c.batch(steps, rev)
	mk, mstep := m.batch(steps, rev)
	checkBatch(t, "chain", status, br, mk, mstep, m)

	// The whole chain is one commit: revision moves 2 -> 3 exactly, and the
	// response agrees with the published statistics.
	if br.Revision != rev+1 {
		t.Fatalf("revision=%d, want exactly one bump to %d", br.Revision, rev+1)
	}
	stF := c.statsT(t)
	if stF.Revision != br.Revision {
		t.Fatalf("response revision %d != stats revision %d", br.Revision, stF.Revision)
	}
	// Physical blocks: src{A,B'}, dst{A',E}, meta/meta2 share D => 5 distinct.
	if stF.UsedBlocks != 5 {
		t.Fatalf("used blocks=%d want 5", stF.UsedBlocks)
	}
	if stF.LogicalFiles != 4 {
		t.Fatalf("files=%d want 4", stF.LogicalFiles)
	}
	// The fork kept both sides of the shared blocks intact.
	if got, s2 := c.read2("dst", 0, 4); s2 != http.StatusOK || !bytes.Equal(got, []byte("FORK")) {
		t.Fatalf("dst head=%q status=%d", got, s2)
	}
	if got, s2 := c.read2("src", BlockSize, 8); s2 != http.StatusOK || !bytes.Equal(got, bytesFill(8, 0x22)) {
		t.Fatalf("src fork=% x status=%d", got, s2)
	}
	if got, s2 := c.read2("meta2", 0, -1); s2 != http.StatusOK || !bytes.Equal(got, []byte("m1")) {
		t.Fatalf("meta2=%q status=%d", got, s2)
	}
	checkStats(t, c, m, "final")
}

// TestBatchDeleteThenCreateHTTP deletes a cloned file and re-creates the same
// name inside one batch. The re-create must observe the staged delete, and
// the surviving clone must keep the shared blocks — and its data — alive.
func TestBatchDeleteThenCreateHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	rev := int64(0)
	nr, st := c.create("old", rev)
	mustOK(t, "create old", m.create("old", rev), st)
	rev = nr
	orig := bytesFill(2*BlockSize, 0xAA)
	nr, st = c.write("old", 0, orig, rev)
	mustOK(t, "write old", m.write("old", 0, orig, rev), st)
	rev = nr
	nr, st = c.clone("old", "keep", rev)
	mustOK(t, "clone keep", m.clone("old", "keep", rev), st)
	rev = nr

	steps := []Step{
		{Op: OpDelete, Name: "old"},
		{Op: OpCreate, Name: "old"}, // would be ErrExists without the staged delete
		{Op: OpWrite, Name: "old", Offset: 0, Data: []byte("NEW!")},
		{Op: OpWrite, Name: "old", Offset: 3 * BlockSize, Data: []byte("tail")}, // sparse hole in between
	}
	status, br := c.batch(steps, rev)
	mk, mstep := m.batch(steps, rev)
	checkBatch(t, "delete then create", status, br, mk, mstep, m)
	if br.Revision != rev+1 {
		t.Fatalf("revision=%d, want exactly one bump to %d", br.Revision, rev+1)
	}
	rev = br.Revision

	// The clone still serves the original bytes from the shared blocks.
	if got, s2 := c.read2("keep", 0, -1); s2 != http.StatusOK || !bytes.Equal(got, orig) {
		t.Fatalf("keep corrupted: len=%d status=%d", len(got), s2)
	}
	// The re-created file is fresh and sparse: NEW! ... hole ... tail.
	want := make([]byte, 3*BlockSize+4)
	copy(want, "NEW!")
	copy(want[3*BlockSize:], "tail")
	if got, s2 := c.read2("old", 0, -1); s2 != http.StatusOK || !bytes.Equal(got, want) {
		t.Fatalf("recreated old mismatch: len=%d status=%d", len(got), s2)
	}
	stF := c.statsT(t)
	if stF.UsedBlocks != 4 { // keep's two shared blocks + two fresh ones
		t.Fatalf("used blocks=%d want 4", stF.UsedBlocks)
	}
	if stF.Revision != rev {
		t.Fatalf("stats revision=%d want %d", stF.Revision, rev)
	}
	checkStats(t, c, m, "final")
}

// TestBatchQuotaBoundaryHTTP exercises the physical-block quota at its exact
// boundary: a batch that precisely consumes the remaining blocks succeeds, a
// batch asking for one block more fails mid-way and rolls back completely,
// and blocks freed by an earlier step can be reused by later steps.
func TestBatchQuotaBoundaryHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	rev := int64(0)
	nr, st := c.create("q", rev)
	mustOK(t, "create q", m.create("q", rev), st)
	rev = nr
	big := bytesFill((MaxBlocks-2)*BlockSize, 0x11)
	nr, st = c.write("q", 0, big, rev)
	mustOK(t, "fill volume", m.write("q", 0, big, rev), st)
	rev = nr

	// Boundary success: exactly the two remaining free blocks.
	steps := []Step{
		{Op: OpCreate, Name: "f"},
		{Op: OpWrite, Name: "f", Offset: 0, Data: bytesFill(2*BlockSize, 0x22)},
	}
	status, br := c.batch(steps, rev)
	mk, mstep := m.batch(steps, rev)
	checkBatch(t, "exact fit", status, br, mk, mstep, m)
	rev = br.Revision
	if got := c.statsT(t).UsedBlocks; got != MaxBlocks {
		t.Fatalf("used=%d want %d (volume exactly full)", got, MaxBlocks)
	}

	// One block too many: step 1 cannot be reserved, and step 0's create
	// must be rolled back with it.
	steps = []Step{
		{Op: OpCreate, Name: "g"},
		{Op: OpWrite, Name: "g", Offset: 0, Data: []byte{1}},
	}
	status, br = c.batch(steps, rev)
	mk, mstep = m.batch(steps, rev)
	checkBatch(t, "over quota", status, br, mk, mstep, m)
	if status != http.StatusInsufficientStorage || br.Step != 1 {
		t.Fatalf("over quota: status=%d step=%d, want 507 step 1", status, br.Step)
	}
	if _, s2 := c.read2("g", 0, -1); s2 != http.StatusNotFound {
		t.Fatalf("rolled-back create is visible: status=%d", s2)
	}
	if stQ := c.statsT(t); stQ.UsedBlocks != MaxBlocks || stQ.Revision != rev {
		t.Fatalf("failed batch changed state: %+v want used=%d rev=%d", stQ, MaxBlocks, rev)
	}

	// Blocks freed by an earlier step are available to later steps of the
	// same batch: delete the huge file, then allocate three fresh blocks.
	steps = []Step{
		{Op: OpDelete, Name: "q"},
		{Op: OpCreate, Name: "h"},
		{Op: OpWrite, Name: "h", Offset: 0, Data: bytesFill(3*BlockSize, 0x33)},
	}
	status, br = c.batch(steps, rev)
	mk, mstep = m.batch(steps, rev)
	checkBatch(t, "free then alloc", status, br, mk, mstep, m)
	rev = br.Revision
	if stF := c.statsT(t); stF.UsedBlocks != 5 || stF.Revision != rev {
		t.Fatalf("after free+alloc: %+v want used=5 rev=%d", stF, rev)
	}
	checkStats(t, c, m, "final")
}

// TestBatchMidFailureRollbackHTTP fails a batch in the middle and verifies
// the reported step index and reason, and that the committed volume — file
// mappings, contents, block pool and revision — is bit-for-bit unchanged.
func TestBatchMidFailureRollbackHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	rev := int64(0)
	nr, st := c.create("a", rev)
	mustOK(t, "create a", m.create("a", rev), st)
	rev = nr
	nr, st = c.write("a", 0, bytesFill(BlockSize, 0x5A), rev)
	mustOK(t, "write a", m.write("a", 0, bytesFill(BlockSize, 0x5A), rev), st)
	rev = nr
	nr, st = c.create("b", rev)
	mustOK(t, "create b", m.create("b", rev), st)
	rev = nr

	before := c.statsT(t)

	// Step 4 references a file that does not exist; steps 0-3 must vanish.
	steps := []Step{
		{Op: OpWrite, Name: "a", Offset: 0, Data: []byte("XX")},
		{Op: OpCreate, Name: "c"},
		{Op: OpWrite, Name: "c", Offset: 0, Data: []byte("data")},
		{Op: OpClone, Src: "a", Dst: "d"},
		{Op: OpWrite, Name: "ghost", Offset: 0, Data: []byte("boom")},
	}
	status, br := c.batch(steps, rev)
	mk, mstep := m.batch(steps, rev)
	checkBatch(t, "mid failure", status, br, mk, mstep, m)
	if status != http.StatusNotFound || br.Step != 4 || !strings.Contains(br.Error, "not found") {
		t.Fatalf("mid failure: status=%d step=%d err=%q, want 404 step 4 not found", status, br.Step, br.Error)
	}

	// The committed state is exactly as before the failed batch.
	if after := c.statsT(t); !reflect.DeepEqual(before, after) {
		t.Fatalf("failed batch changed stats\nbefore=%+v\nafter =%+v", before, after)
	}
	if got, s2 := c.read2("a", 0, 4); s2 != http.StatusOK || !bytes.Equal(got, bytesFill(4, 0x5A)) {
		t.Fatalf("a was modified by rolled-back batch: % x", got)
	}
	if _, s2 := c.read2("c", 0, -1); s2 != http.StatusNotFound {
		t.Fatalf("staged create c visible: %d", s2)
	}
	if _, s2 := c.read2("d", 0, -1); s2 != http.StatusNotFound {
		t.Fatalf("staged clone d visible: %d", s2)
	}

	// An exists failure reports its step index (>= 0), unlike a revision
	// conflict which is batch-level (step -1).
	dup := []Step{{Op: OpCreate, Name: "a"}}
	status, br = c.batch(dup, rev)
	mk, mstep = m.batch(dup, rev)
	checkBatch(t, "exists", status, br, mk, mstep, m)
	if br.Step != 0 || !strings.Contains(br.Error, "exists") {
		t.Fatalf("exists: step=%d err=%q, want step 0 exists", br.Step, br.Error)
	}
	stale := []Step{{Op: OpCreate, Name: "t"}}
	status, br = c.batch(stale, rev+3)
	mk, mstep = m.batch(stale, rev+3)
	checkBatch(t, "stale revision", status, br, mk, mstep, m)
	if br.Step != -1 || !strings.Contains(br.Error, "conflict") {
		t.Fatalf("stale: step=%d err=%q, want step -1 conflict", br.Step, br.Error)
	}
	checkStats(t, c, m, "final")
}

// TestBatchConcurrentSameRevisionHTTP races two different batches carrying
// the same expected revision: exactly one commits, the loser leaves no trace.
func TestBatchConcurrentSameRevisionHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()
	if _, st := c.create("base", 0); st != http.StatusOK {
		t.Fatal(st)
	}
	m.create("base", 0)

	batchX := []Step{{Op: OpCreate, Name: "x"}, {Op: OpWrite, Name: "x", Offset: 0, Data: []byte("X")}}
	batchY := []Step{{Op: OpCreate, Name: "y"}, {Op: OpWrite, Name: "y", Offset: 0, Data: []byte("Y")}}

	const n = 64
	var wg sync.WaitGroup
	codes := make([]int, n)
	steps := make([]int, n)
	revisions := make([]int64, n)
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			b := batchX
			if i%2 == 1 {
				b = batchY
			}
			status, br := c.batch(b, 1)
			codes[i], steps[i], revisions[i] = status, br.Step, br.Revision
		}(i)
	}
	wg.Wait()

	ok := 0
	for i := range codes {
		switch codes[i] {
		case http.StatusOK:
			ok++
			if revisions[i] != 2 {
				t.Fatalf("winner revision=%d want 2", revisions[i])
			}
		case http.StatusConflict:
			if steps[i] != -1 {
				t.Fatalf("loser failing step=%d, want batch-level -1", steps[i])
			}
		default:
			t.Fatalf("unexpected status %d", codes[i])
		}
	}
	if ok != 1 {
		t.Fatalf("exactly one batch may win, got %d/%d", ok, n)
	}

	// Only the winner's effects are visible; mirror them in the model.
	st := c.statsT(t)
	winnerIsX := false
	for _, fi := range st.Files {
		if fi.Name == "x" {
			winnerIsX = true
		}
	}
	if winnerIsX {
		m.batch(batchX, 1)
	} else {
		m.batch(batchY, 1)
	}
	if st.Revision != 2 || st.LogicalFiles != 2 || st.UsedBlocks != 1 {
		t.Fatalf("post-race stats: %+v", st)
	}
	checkStats(t, c, m, "post race")
}

// TestBatchValidationHTTP covers batch-level and per-step static rejection:
// shape limits, unknown operations, bad names and ranges, revision sourcing,
// and the zero-length-write no-op inside a batch.
func TestBatchValidationHTTP(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()
	if _, st := c.create("seed", 0); st != http.StatusOK {
		t.Fatal(st)
	}
	m.create("seed", 0)

	nineValid := make([]Step, MaxBatchSteps+1)
	for i := range nineValid {
		nineValid[i] = Step{Op: OpCreate, Name: fmt.Sprintf("n%d", i)}
	}
	cases := []struct {
		desc  string
		steps []Step
		rev   int64
	}{
		{"empty batch", nil, 1},
		{"too many steps", nineValid, 1},
		{"unknown op", []Step{{Op: OpCreate, Name: "t1"}, {Op: "rename", Name: "t2"}, {Op: OpCreate, Name: "t3"}}, 1},
		{"bad name at step 2", []Step{{Op: OpCreate, Name: "t1"}, {Op: OpCreate, Name: "t2"}, {Op: OpCreate, Name: "bad name"}}, 1},
		{"negative offset", []Step{{Op: OpCreate, Name: "t1"}, {Op: OpWrite, Name: "t1", Offset: -1, Data: []byte("x")}}, 1},
		{"negative length", []Step{{Op: OpCreate, Name: "t1"}, {Op: OpTruncate, Name: "t1", Length: -5}}, 1},
		{"zero write past EOF", []Step{{Op: OpCreate, Name: "z"}, {Op: OpWrite, Name: "z", Offset: 5}}, 1},
		{"stale revision", []Step{{Op: OpCreate, Name: "t1"}}, 7},
		{"zero write no-op", []Step{{Op: OpCreate, Name: "z2"}, {Op: OpWrite, Name: "z2", Offset: 0}}, 1},
	}
	for _, tc := range cases {
		status, br := c.batch(tc.steps, tc.rev)
		mk, mstep := m.batch(tc.steps, tc.rev)
		checkBatch(t, tc.desc, status, br, mk, mstep, m)
	}
	// Only the last case committed; it bumped the revision exactly once.
	if m.revision != 2 {
		t.Fatalf("model revision=%d want 2", m.revision)
	}

	// Missing revision everywhere is a 400 and changes nothing.
	body, _ := json.Marshal(map[string]any{"steps": []Step{{Op: OpCreate, Name: "tq"}}})
	resp, err := c.hc.Post(c.srv.URL+"/batch", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("missing revision status=%d", resp.StatusCode)
	}
	resp.Body.Close()

	// Malformed JSON is a 400.
	resp, err = c.hc.Post(c.srv.URL+"/batch", "application/json", strings.NewReader("{nope"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("malformed body status=%d", resp.StatusCode)
	}
	resp.Body.Close()

	// GET on the mutation endpoint is a 405.
	resp, err = c.hc.Get(c.srv.URL + "/batch")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /batch status=%d", resp.StatusCode)
	}
	resp.Body.Close()

	// The rev query parameter (and header) take precedence over the body.
	body, _ = json.Marshal(map[string]any{"revision": 9999, "steps": []Step{{Op: OpCreate, Name: "tq"}}})
	resp, err = c.hc.Post(c.srv.URL+"/batch?rev=2", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	br := decodeBatchResp(resp)
	if resp.StatusCode != http.StatusOK || br.Revision != 3 {
		t.Fatalf("query rev precedence: status=%d rev=%d", resp.StatusCode, br.Revision)
	}
	mk, _ := m.batch([]Step{{Op: OpCreate, Name: "tq"}}, 2)
	if mk != kOK {
		t.Fatalf("model query precedence: %v", mk)
	}

	checkStats(t, c, m, "final")
}

// TestBatchAtomicVisibilityHTTP hammers the volume with alternating
// create-all / delete-all batches while readers sample /stats: every observed
// snapshot must be the complete pre- or post-commit state, never a mixture.
func TestBatchAtomicVisibilityHTTP(t *testing.T) {
	c := newAPIClient(t)
	if _, st := c.create("base", 0); st != http.StatusOK {
		t.Fatal(st)
	}

	payload := []byte("12345678")
	batchAdd := []Step{
		{Op: OpCreate, Name: "g1"}, {Op: OpWrite, Name: "g1", Offset: 0, Data: payload},
		{Op: OpCreate, Name: "g2"}, {Op: OpWrite, Name: "g2", Offset: 0, Data: payload},
		{Op: OpCreate, Name: "g3"}, {Op: OpWrite, Name: "g3", Offset: 0, Data: payload},
	}
	batchDel := []Step{
		{Op: OpDelete, Name: "g1"},
		{Op: OpDelete, Name: "g2"},
		{Op: OpDelete, Name: "g3"},
	}

	var stop atomic.Bool
	var violations atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				resp, err := c.hc.Get(c.srv.URL + "/stats")
				if err != nil {
					return
				}
				var st Stats
				_ = json.NewDecoder(resp.Body).Decode(&st)
				resp.Body.Close()
				g := 0
				for _, fi := range st.Files {
					if fi.Name == "g1" || fi.Name == "g2" || fi.Name == "g3" {
						g++
					}
				}
				// All three files with their blocks, or none of them.
				if g != 0 && g != 3 {
					violations.Add(1)
				}
				if st.UsedBlocks != g || st.LogicalFiles != 1+g {
					violations.Add(1)
				}
			}
		}()
	}

	rev := int64(1)
	const rounds = 60
	for i := 0; i < rounds; i++ {
		status, br := c.batch(batchAdd, rev)
		if status != http.StatusOK {
			t.Fatalf("add batch: status=%d err=%q step=%d", status, br.Error, br.Step)
		}
		rev = br.Revision
		status, br = c.batch(batchDel, rev)
		if status != http.StatusOK {
			t.Fatalf("del batch: status=%d err=%q step=%d", status, br.Error, br.Step)
		}
		rev = br.Revision
	}
	stop.Store(true)
	wg.Wait()
	if n := violations.Load(); n != 0 {
		t.Fatalf("readers observed %d partially committed snapshots", n)
	}
	if rev != 1+2*rounds {
		t.Fatalf("revision=%d want %d (one bump per batch)", rev, 1+2*rounds)
	}
}

// TestBatchInProcess drives Volume.Batch directly: failing-step reporting,
// revision accounting, and proof that a failed batch leaves the real free
// block pool untouched.
func TestBatchInProcess(t *testing.T) {
	v := New()
	rev, err := v.Create("f1", 0)
	if err != nil {
		t.Fatal(err)
	}
	if rev, err = v.Write("f1", 0, []byte("A"), rev); err != nil { // block 0
		t.Fatal(err)
	}
	if rev, err = v.Create("f2", rev); err != nil {
		t.Fatal(err)
	}
	if rev, err = v.Write("f2", 0, []byte("B"), rev); err != nil { // block 1
		t.Fatal(err)
	}
	if rev, err = v.Delete("f1", rev); err != nil { // frees block 0
		t.Fatal(err)
	}
	if id := v.DebugBlockID("f2", 0); id != 1 {
		t.Fatalf("f2 block 0 id=%d want 1", id)
	}

	// The staged delete frees f2's block inside the snapshot, then the write
	// fails: the real volume and its free list must be untouched.
	nr, step, err := v.Batch([]Step{
		{Op: OpDelete, Name: "f2"},
		{Op: OpWrite, Name: "f2", Offset: 0, Data: []byte("x")},
	}, rev)
	if !errors.Is(err, ErrNotFound) || step != 1 || nr != rev {
		t.Fatalf("batch: rev=%d step=%d err=%v, want rev=%d step=1 ErrNotFound", nr, step, err, rev)
	}
	if id := v.DebugBlockID("f2", 0); id != 1 {
		t.Fatalf("failed batch unmapped f2 block: id=%d", id)
	}
	if v.Revision() != rev {
		t.Fatalf("failed batch bumped revision to %d", v.Revision())
	}
	if st := v.Stats(); st.UsedBlocks != 1 || st.LogicalFiles != 1 {
		t.Fatalf("failed batch changed stats: %+v", st)
	}
	// The next allocation must pop block 0 — the only genuinely free block.
	// Had the staged free of block 1 leaked into the real free list, this
	// write would reuse the still-mapped block 1.
	if rev, err = v.Write("f2", BlockSize, []byte("y"), rev); err != nil {
		t.Fatal(err)
	}
	if id := v.DebugBlockID("f2", 1); id != 0 {
		t.Fatalf("free pool corrupted by failed batch: new block id=%d want 0", id)
	}

	// A five-step batch commits once: the revision advances by exactly one.
	nr, step, err = v.Batch([]Step{
		{Op: OpCreate, Name: "t1"},
		{Op: OpCreate, Name: "t2"},
		{Op: OpDelete, Name: "t1"},
		{Op: OpWrite, Name: "f2", Offset: 0, Data: []byte("z")},
		{Op: OpDelete, Name: "t2"},
	}, rev)
	if err != nil || step != -1 || nr != rev+1 {
		t.Fatalf("batch: rev=%d step=%d err=%v, want rev=%d", nr, step, err, rev+1)
	}
	if v.Revision() != rev+1 {
		t.Fatalf("revision=%d, want exactly one bump to %d", v.Revision(), rev+1)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 2 || st.LogicalFiles != 1 {
		t.Fatalf("create+delete inside batch leaked: %+v", st)
	}
	if got, _ := v.Read("f2", 0, 1); len(got) != 1 || got[0] != 'z' {
		t.Fatalf("f2[0]=%q want z", got)
	}

	// A zero-length write step is a no-op; the batch still commits once.
	nr, step, err = v.Batch([]Step{
		{Op: OpCreate, Name: "z"},
		{Op: OpWrite, Name: "z", Offset: 0},
	}, rev)
	if err != nil || step != -1 || nr != rev+1 {
		t.Fatalf("zero-write batch: rev=%d step=%d err=%v", nr, step, err)
	}
	rev = nr
	if st := v.Stats(); st.LogicalFiles != 2 || st.UsedBlocks != 2 {
		t.Fatalf("zero-write batch changed occupancy: %+v", st)
	}

	// Batch-level rejections report step -1 and leave the revision alone.
	if _, step, err = v.Batch(nil, rev); !errors.Is(err, ErrInvalidBatch) || step != -1 {
		t.Fatalf("empty batch: step=%d err=%v", step, err)
	}
	if _, step, err = v.Batch(make([]Step, MaxBatchSteps+1), rev); !errors.Is(err, ErrInvalidBatch) || step != -1 {
		t.Fatalf("oversized batch: step=%d err=%v", step, err)
	}
	if _, step, err = v.Batch([]Step{{Op: "rename", Name: "z"}}, rev); !errors.Is(err, ErrInvalidBatch) || step != 0 {
		t.Fatalf("unknown op: step=%d err=%v", step, err)
	}
	if _, step, err = v.Batch([]Step{{Op: OpCreate, Name: "zz"}}, rev+9); !errors.Is(err, ErrConflict) || step != -1 {
		t.Fatalf("stale batch: step=%d err=%v", step, err)
	}
	if v.Revision() != rev {
		t.Fatalf("rejected batches bumped revision to %d", v.Revision())
	}
}

// ---------------------------------------------------------------------------
// Randomized differential fuzzing of batches over HTTP.
// ---------------------------------------------------------------------------

func TestBatchFuzzHTTP(t *testing.T) {
	for _, seed := range []int64{11, 23, 42} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			fuzzBatch(t, seed)
		})
	}
}

func randBatchStep(rng *rand.Rand, names []string, maxOff int64) Step {
	pick := func() string { return names[rng.Intn(len(names))] }
	op := rng.Intn(100)
	switch {
	case op < 20: // create
		name := pick()
		if rng.Intn(40) == 0 {
			name = "bad name"
		}
		return Step{Op: OpCreate, Name: name}
	case op < 38: // clone
		return Step{Op: OpClone, Src: pick(), Dst: pick()}
	case op < 68: // write
		off := rng.Int63n(maxOff)
		if rng.Intn(50) == 0 {
			off = -1
		}
		size := rng.Intn(2 * BlockSize)
		if rng.Intn(15) == 0 {
			size = 0 // zero-length write no-op
		}
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(rng.Intn(256))
		}
		return Step{Op: OpWrite, Name: pick(), Offset: off, Data: data}
	case op < 84: // truncate
		length := rng.Int63n(maxOff)
		if rng.Intn(60) == 0 {
			length = -1
		}
		return Step{Op: OpTruncate, Name: pick(), Length: length}
	case op < 97: // delete
		return Step{Op: OpDelete, Name: pick()}
	default: // unknown operation
		return Step{Op: Op("bogus"), Name: pick()}
	}
}

func fuzzBatch(t *testing.T, seed int64) {
	c := newAPIClient(t)
	m := newModel()
	rng := rand.New(rand.NewSource(seed))
	names := []string{"a", "b", "c", "d", "e"}
	const iters = 400
	maxOff := int64(6 * BlockSize)

	for it := 0; it < iters; it++ {
		ctx := fmt.Sprintf("seed=%d iter=%d rev=%d", seed, it, m.revision)

		// Occasionally use a stale revision to exercise batch-level conflicts.
		rev := m.revision
		if rng.Intn(8) == 0 {
			rev = rng.Int63n(m.revision + 5)
		}
		steps := make([]Step, 1+rng.Intn(MaxBatchSteps))
		for i := range steps {
			steps[i] = randBatchStep(rng, names, maxOff)
		}
		status, br := c.batch(steps, rev)
		mk, mstep := m.batch(steps, rev)
		checkBatch(t, ctx, status, br, mk, mstep, m)

		// Every batch is all-or-nothing: the committed volume must equal the
		// model whether the batch committed or rolled back.
		if it%3 == 0 {
			checkStats(t, c, m, ctx)
		}
	}
	checkStats(t, c, m, "final")
}
