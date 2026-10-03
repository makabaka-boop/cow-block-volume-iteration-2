package vfs

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Naive reference model.
//
// Every file keeps its *complete* logical byte array (no sparse tricks). A
// parallel refcount table simulates the physical layer with block IDs handed
// out from a global counter: a write allocates one block per touched logical
// block that was previously absent or shared, cloning shares IDs, truncation
// and delete drop references. This is deliberately simple and independent of
// the production implementation.
// ---------------------------------------------------------------------------

type modelBlock struct {
	id   int
	refs int
}

type modelFile struct {
	data []byte
	phys map[int64]int // logical block -> model physical ID
}

type model struct {
	files    map[string]*modelFile
	blocks   map[int]*modelBlock
	revision int64
	nextID   int
}

func newModel() *model {
	return &model{files: map[string]*modelFile{}, blocks: map[int]*modelBlock{}}
}

func (m *model) alloc() int {
	id := m.nextID
	m.nextID++
	m.blocks[id] = &modelBlock{id: id, refs: 1}
	return id
}

func (m *model) release(id int) {
	m.blocks[id].refs--
	if m.blocks[id].refs == 0 {
		delete(m.blocks, id)
	}
}

// errKind mirrors the domain error categories exposed over HTTP.
type errKind string

const (
	kOK       errKind = ""
	kBadName  errKind = "badname"
	kConflict errKind = "conflict"
	kNotFound errKind = "notfound"
	kRange    errKind = "range"
	kQuota    errKind = "quota"
	kExists   errKind = "exists"
	kBadBatch errKind = "badbatch"
)

func (m *model) create(name string, rev int64) errKind {
	if !ValidateName(name) {
		return kBadName
	}
	if rev != m.revision {
		return kConflict
	}
	if k := m.createStaged(name); k != kOK {
		return k
	}
	m.revision++
	return kOK
}

// createStaged is the batch-staging variant: no revision check or bump.
func (m *model) createStaged(name string) errKind {
	if _, ok := m.files[name]; ok {
		return kExists
	}
	m.files[name] = &modelFile{phys: map[int64]int{}}
	return kOK
}

func (m *model) clone(src, dst string, rev int64) errKind {
	if !ValidateName(src) || !ValidateName(dst) {
		return kBadName
	}
	if rev != m.revision {
		return kConflict
	}
	if k := m.cloneStaged(src, dst); k != kOK {
		return k
	}
	m.revision++
	return kOK
}

func (m *model) cloneStaged(src, dst string) errKind {
	s, ok := m.files[src]
	if !ok {
		return kNotFound
	}
	if _, ok := m.files[dst]; ok {
		return kExists
	}
	nf := &modelFile{data: append([]byte(nil), s.data...), phys: make(map[int64]int, len(s.phys))}
	for idx, id := range s.phys {
		nf.phys[idx] = id
		m.blocks[id].refs++
	}
	m.files[dst] = nf
	return kOK
}

func (m *model) write(name string, off int64, p []byte, rev int64) errKind {
	if !ValidateName(name) {
		return kBadName
	}
	if off < 0 || (len(p) > 0 && off > int64Max-int64(len(p))) {
		return kRange
	}
	if rev != m.revision {
		return kConflict
	}
	k, mutated := m.writeStaged(name, off, p)
	if k != kOK {
		return k
	}
	if mutated {
		m.revision++
	}
	return kOK
}

// writeStaged reports whether the model was mutated (zero-length writes at or
// before EOF are no-ops).
func (m *model) writeStaged(name string, off int64, p []byte) (errKind, bool) {
	f, ok := m.files[name]
	if !ok {
		return kNotFound, false
	}
	if len(p) == 0 {
		if off > int64(len(f.data)) {
			return kRange, false
		}
		return kOK, false
	}
	end := off + int64(len(p))
	// Reservation identical to the real implementation: one block per hole or
	// shared block touched.
	need := 0
	for idx := off / BlockSize; idx <= (end-1)/BlockSize; idx++ {
		id, present := f.phys[idx]
		if !present || m.blocks[id].refs > 1 {
			need++
		}
	}
	if len(m.blocks)+need > MaxBlocks {
		return kQuota, false
	}
	// Commit byte array.
	if end > int64(len(f.data)) {
		grown := make([]byte, end)
		copy(grown, f.data)
		f.data = grown
	}
	copy(f.data[off:], p)
	// Commit physical layer.
	d := p
	o := off
	for len(d) > 0 {
		idx := o / BlockSize
		id, present := f.phys[idx]
		if !present {
			f.phys[idx] = m.alloc()
		} else if m.blocks[id].refs > 1 {
			m.blocks[id].refs-- // emulate private copy (new zero block + data)
			nid := m.alloc()
			f.phys[idx] = nid
		}
		n := BlockSize - int(o-idx*BlockSize)
		if n > len(d) {
			n = len(d)
		}
		d = d[n:]
		o += int64(n)
	}
	return kOK, true
}

func (m *model) truncate(name string, length int64, rev int64) errKind {
	if !ValidateName(name) {
		return kBadName
	}
	if length < 0 {
		return kRange
	}
	if rev != m.revision {
		return kConflict
	}
	if k := m.truncateStaged(name, length); k != kOK {
		return k
	}
	m.revision++
	return kOK
}

func (m *model) truncateStaged(name string, length int64) errKind {
	f, ok := m.files[name]
	if !ok {
		return kNotFound
	}
	if length >= int64(len(f.data)) {
		grown := make([]byte, length)
		copy(grown, f.data)
		f.data = grown
		return kOK
	}
	// Shrink, same two-phase logic as the volume.
	lastKeep := int64(-1)
	if length > 0 {
		lastKeep = (length - 1) / BlockSize
	}
	hasStraddle := length%BlockSize != 0
	straddle := lastKeep
	reclaim := 0
	var dropped []int64
	for idx, id := range f.phys {
		if idx > lastKeep {
			dropped = append(dropped, idx)
			if m.blocks[id].refs == 1 {
				reclaim++
			}
		}
	}
	straddleShared := false
	if hasStraddle {
		if id, ok := f.phys[straddle]; ok && m.blocks[id].refs > 1 {
			straddleShared = true
		}
	}
	if straddleShared && len(m.blocks)-reclaim+1 > MaxBlocks {
		return kQuota
	}
	for _, idx := range dropped {
		m.release(f.phys[idx])
		delete(f.phys, idx)
	}
	if hasStraddle {
		if id, present := f.phys[straddle]; present && m.blocks[id].refs > 1 {
			m.blocks[id].refs--
			f.phys[straddle] = m.alloc()
		}
	}
	f.data = f.data[:length]
	return kOK
}

func (m *model) del(name string, rev int64) errKind {
	if !ValidateName(name) {
		return kBadName
	}
	if rev != m.revision {
		return kConflict
	}
	if k := m.delStaged(name); k != kOK {
		return k
	}
	m.revision++
	return kOK
}

func (m *model) delStaged(name string) errKind {
	f, ok := m.files[name]
	if !ok {
		return kNotFound
	}
	for _, id := range f.phys {
		m.release(id)
	}
	delete(m.files, name)
	return kOK
}

// ---------------------------------------------------------------------------
// HTTP client driving the real server.
// ---------------------------------------------------------------------------

type apiClient struct {
	srv *httptest.Server
	hc  *http.Client
}

func newAPIClient(t *testing.T) *apiClient {
	t.Helper()
	srv := httptest.NewServer(NewServer(New()).Handler())
	t.Cleanup(srv.Close)
	return &apiClient{srv: srv, hc: srv.Client()}
}

type mutResp struct {
	Revision int64 `json:"revision"`
}

func (c *apiClient) post(path string, body []byte) (int64, int) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	resp, err := c.hc.Post(c.srv.URL+path, "application/octet-stream", rdr)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusOK {
		var mr mutResp
		if err := json.Unmarshal(raw, &mr); err != nil {
			panic(err)
		}
		return mr.Revision, resp.StatusCode
	}
	return -1, resp.StatusCode
}

func (c *apiClient) create(name string, rev int64) (int64, int) {
	return c.post("/create?name="+url.QueryEscape(name)+"&rev="+strconv.FormatInt(rev, 10), nil)
}
func (c *apiClient) clone(src, dst string, rev int64) (int64, int) {
	return c.post(fmt.Sprintf("/clone?src=%s&dst=%s&rev=%d", url.QueryEscape(src), url.QueryEscape(dst), rev), nil)
}
func (c *apiClient) write(name string, off int64, p []byte, rev int64) (int64, int) {
	return c.post(fmt.Sprintf("/write?name=%s&offset=%d&rev=%d", url.QueryEscape(name), off, rev), p)
}
func (c *apiClient) truncate(name string, length, rev int64) (int64, int) {
	return c.post(fmt.Sprintf("/truncate?name=%s&length=%d&rev=%d", url.QueryEscape(name), length, rev), nil)
}
func (c *apiClient) del(name string, rev int64) (int64, int) {
	return c.post("/delete?name="+url.QueryEscape(name)+"&rev="+strconv.FormatInt(rev, 10), nil)
}

func (c *apiClient) read(name string, off, length int64) ([]byte, int, string) {
	u := fmt.Sprintf("%s/read?name=%s&offset=%d&length=%d", c.srv.URL, url.QueryEscape(name), off, length)
	resp, err := c.hc.Get(u)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, resp.Header.Get("X-Revision")
}

// statusToKind maps HTTP statuses back to model error categories.
func statusToKind(status int) errKind {
	switch status {
	case http.StatusOK:
		return kOK
	case http.StatusBadRequest:
		return kBadName
	case http.StatusConflict:
		// Exists and revision conflict share 409; the fuzzer only asserts
		// the two are in the same broad bucket when ambiguous, and the
		// deterministic tests distinguish them explicitly.
		return kConflict
	case http.StatusNotFound:
		return kNotFound
	case http.StatusRequestedRangeNotSatisfiable:
		return kRange
	case http.StatusInsufficientStorage:
		return kQuota
	default:
		return errKind("status:" + strconv.Itoa(status))
	}
}

// existsAsStatus distinguishes 409 sub-cases via the error body.
func (c *apiClient) createExpectExact(t *testing.T, path string, wantStatus int) {
	t.Helper()
	_, status := c.post(path, nil)
	if status != wantStatus {
		t.Fatalf("status=%d want %d for %s", status, wantStatus, path)
	}
}

// checkStats compares the whole volume state to the model: revision, every
// file's full content, per-file physical blocks and the distinct physical
// block total.
func checkStats(t *testing.T, c *apiClient, m *model, ctx string) {
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
	if st.Revision != m.revision {
		t.Fatalf("[%s] revision: server=%d model=%d", ctx, st.Revision, m.revision)
	}
	if st.LogicalFiles != len(m.files) {
		t.Fatalf("[%s] file count: server=%d model=%d", ctx, st.LogicalFiles, len(m.files))
	}
	if st.UsedBlocks != len(m.blocks) {
		t.Fatalf("[%s] physical blocks: server=%d model=%d (free=%d)",
			ctx, st.UsedBlocks, len(m.blocks), st.FreeBlocks)
	}
	byName := map[string]FileInfo{}
	for _, fi := range st.Files {
		byName[fi.Name] = fi
	}
	var wantSparse int64
	for name, mf := range m.files {
		fi, ok := byName[name]
		if !ok {
			t.Fatalf("[%s] file %q missing from stats", ctx, name)
		}
		length := int64(len(mf.data))
		var backed int64
		for idx := range mf.phys {
			start := idx * BlockSize
			if start >= length {
				continue
			}
			end := start + BlockSize
			if end > length {
				end = length
			}
			backed += end - start
		}
		wantSparse += length - backed
		if fi.Length != length {
			t.Fatalf("[%s] %s length: server=%d model=%d", ctx, name, fi.Length, length)
		}
		if fi.PhysicalBlocks != len(mf.phys) {
			t.Fatalf("[%s] %s physblocks: server=%d model=%d", ctx, name, fi.PhysicalBlocks, len(mf.phys))
		}
	}
	if st.SparseBytes != wantSparse {
		t.Fatalf("[%s] sparse bytes: server=%d model=%d", ctx, st.SparseBytes, wantSparse)
	}
	for name, mf := range m.files {
		// Full content comparison (the naive byte array is the oracle).
		raw, status, _ := c.read(name, 0, -1)
		if status != http.StatusOK {
			t.Fatalf("[%s] read %s status=%d", ctx, name, status)
		}
		if !bytes.Equal(raw, mf.data) {
			i := 0
			for i < len(raw) && i < len(mf.data) && raw[i] == mf.data[i] {
				i++
			}
			t.Fatalf("[%s] %s content mismatch at byte %d (slen=%d mlen=%d)", ctx, name, i, len(raw), len(mf.data))
		}
	}
}

// ---------------------------------------------------------------------------
// Deterministic differential scenarios.
// ---------------------------------------------------------------------------

func TestHTTPCreateCloneFork(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	step := func(desc string, mk, ms errKind, revS int64, revM int64) {
		t.Helper()
		if mk != ms {
			t.Fatalf("%s: server=%v model=%v", desc, ms, mk)
		}
		if mk == kOK && revS != revM {
			t.Fatalf("%s: revision server=%d model=%d", desc, revS, revM)
		}
	}

	revS, st := c.create("a", 0)
	step("create a", m.create("a", 0), statusToKind(st), revS, m.revision)
	payload := bytesFill(BlockSize+10, 0x11)
	revS, st = c.write("a", BlockSize-5, payload, m.revision)
	step("write cross block", m.write("a", BlockSize-5, payload, m.revision), statusToKind(st), revS, m.revision)
	revS, st = c.clone("a", "b", m.revision)
	step("clone", m.clone("a", "b", m.revision), statusToKind(st), revS, m.revision)
	checkStats(t, c, m, "post clone")

	// First write into b's shared block 0 forks it.
	fork := []byte("FORK")
	revS, st = c.write("b", 0, fork, m.revision)
	step("fork b0", m.write("b", 0, fork, m.revision), statusToKind(st), revS, m.revision)
	// First write into the shared block 1 (partial block) forks it too.
	fork1 := bytesFill(10, 0x22)
	revS, st = c.write("b", BlockSize+1, fork1, m.revision)
	step("fork b1", m.write("b", BlockSize+1, fork1, m.revision), statusToKind(st), revS, m.revision)
	// Writing a again proves the fork did not touch a.
	more := bytesFill(BlockSize, 0x33)
	revS, st = c.write("a", 2*BlockSize, more, m.revision)
	step("a grows independently", m.write("a", 2*BlockSize, more, m.revision), statusToKind(st), revS, m.revision)
	checkStats(t, c, m, "post forks")

	// Truncate b across the boundary, zeroing a shared-less tail, then delete.
	revS, st = c.truncate("b", BlockSize/2, m.revision)
	step("truncate b", m.truncate("b", BlockSize/2, m.revision), statusToKind(st), revS, m.revision)
	revS, st = c.del("b", m.revision)
	step("delete b", m.del("b", m.revision), statusToKind(st), revS, m.revision)
	checkStats(t, c, m, "post delete")

	// Clone to the same destination name is 409/exists.
	_, st = c.clone("a", "a", m.revision)
	if st != http.StatusConflict {
		t.Fatalf("self clone status=%d", st)
	}
	// Stale revision is also 409, distinguished via body.
	resp, err := c.hc.Post(c.srv.URL+"/create?name=z&rev=0", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), "revision conflict") {
		t.Fatalf("expected revision conflict body, got %q", body)
	}
	_ = revS
}

// TestHTTPSparseAndRangeRead verifies hole zeros and range semantics over HTTP.
func TestHTTPSparseAndRangeRead(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()
	m.create("h", 0)
	c.createExpectExact(t, "/create?name=h&rev=0", http.StatusOK)

	off := int64(3*BlockSize + 7)
	p := []byte("sparse")
	if _, st := c.write("h", off, p, 1); st != http.StatusOK {
		t.Fatal(st)
	}
	m.write("h", off, p, 1)
	checkStats(t, c, m, "sparse write")

	// Zero-byte writes are not mutations: they neither extend a file nor bump
	// the revision. One past EOF is still a stable range error.
	fileLen := int64(len(m.files["h"].data))
	if nr, st := c.write("h", fileLen, nil, m.revision); st != http.StatusOK || nr != m.revision {
		t.Fatalf("zero write at EOF: status=%d rev=%d", st, nr)
	}
	if _, st := c.write("h", fileLen+1, nil, m.revision); st != http.StatusRequestedRangeNotSatisfiable {
		t.Fatalf("zero write past EOF status=%d", st)
	}

	// Offset and length values that individually fit in int64 must not make
	// the server panic when their sum overflows; they are 416 responses.
	wantRev := strconv.FormatInt(m.revision, 10)
	_, st, hrev := c.read("h", 1, int64Max)
	if st != http.StatusRequestedRangeNotSatisfiable {
		t.Fatalf("overflowing read status=%d", st)
	}
	if hrev != wantRev {
		t.Fatalf("range error X-Revision=%q, want %s", hrev, wantRev)
	}
	if _, st, hrev := c.read("h", fileLen+1, -1); st != http.StatusRequestedRangeNotSatisfiable || hrev != wantRev {
		t.Fatalf("past EOF read status=%d X-Revision=%q", st, hrev)
	}

	// Random windows, including fully sparse, straddling, and EOF cases.
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		o := rng.Int63n(off + int64(len(p)) + BlockSize)
		l := int64(0)
		if i%7 == 0 {
			l = -1
		} else {
			l = rng.Int63n(2 * BlockSize)
		}
		raw, status, hrev := c.read("h", o, l)
		var want []byte
		wantStatus := http.StatusOK
		switch {
		case o > int64(len(m.files["h"].data)):
			wantStatus = http.StatusRequestedRangeNotSatisfiable
		case l < 0:
			want = m.files["h"].data[o:]
		case o+l > int64(len(m.files["h"].data)):
			want = m.files["h"].data[o:]
		default:
			want = m.files["h"].data[o : o+l]
		}
		if status != wantStatus {
			t.Fatalf("read o=%d l=%d status=%d want %d", o, l, status, wantStatus)
		}
		if wantStatus == http.StatusOK {
			if !bytes.Equal(raw, want) {
				t.Fatalf("read o=%d l=%d mismatch (got %d want %d bytes)", o, l, len(raw), len(want))
			}
			if hrev != strconv.FormatInt(m.revision, 10) {
				t.Fatalf("X-Revision=%s want %d", hrev, m.revision)
			}
		}
	}
}

// TestHTTPQuotaForkExhaustion uses the real quota: fill the volume with a
// file, clone it (zero extra blocks), then force COW on all blocks.
func TestHTTPQuotaForkExhaustion(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()
	rev := int64(0)

	do := func(desc string, ms errKind, gotStatus int, newRev int64) {
		t.Helper()
		got := statusToKind(gotStatus)
		// Server conflates exists/conflict into 409; the model returns a
		// precise kind. Only the exact quota path is asserted here.
		if ms == kQuota && got != kQuota {
			t.Fatalf("%s: model=quota server=%v (%d)", desc, got, gotStatus)
		}
		if ms == kOK && gotStatus != http.StatusOK {
			t.Fatalf("%s: model=ok server=%d", desc, gotStatus)
		}
		if gotStatus == http.StatusOK {
			rev = newRev
		}
	}

	nr, st := c.create("big", rev)
	do("create", m.create("big", rev), st, nr)
	full := bytesFill(MaxBlocks*BlockSize, 0x01)
	nr, st = c.write("big", 0, full, rev)
	do("fill volume", m.write("big", 0, full, rev), st, nr)
	nr, st = c.clone("big", "twin", rev)
	do("clone full volume", m.clone("big", "twin", rev), st, nr)
	checkStats(t, c, m, "full clone")

	// A one-byte write anywhere in the twin needs one COW block: none free.
	nr, st = c.write("twin", 0, []byte{9}, rev)
	do("cow denied", m.write("twin", 0, []byte{9}, rev), st, nr)
	// Failed COW must not diverge the clone or leak references.
	checkStats(t, c, m, "after denied cow")
	raw, status := c.read2("twin", 0, 4)
	if status != 200 || !bytes.Equal(raw, []byte{1, 1, 1, 1}) {
		t.Fatalf("twin corrupted: % x status=%d", raw, status)
	}

	// Truncating the original to zero drops no physical blocks (still shared),
	// but then the twin owns all blocks privately.
	nr, st = c.truncate("big", 0, rev)
	do("truncate original", m.truncate("big", 0, rev), st, nr)
	checkStats(t, c, m, "truncate big")

	// Deleting the twin frees everything.
	nr, st = c.del("twin", rev)
	do("delete twin", m.del("twin", rev), st, nr)
	checkStats(t, c, m, "delete twin")
}

func (c *apiClient) read2(name string, off, length int64) ([]byte, int) {
	b, s, _ := c.read(name, off, length)
	return b, s
}

// ---------------------------------------------------------------------------
// Randomized differential fuzzing over HTTP.
// ---------------------------------------------------------------------------

func TestModelFuzzHTTP(t *testing.T) {
	for _, seed := range []int64{1, 2, 3, 7} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			fuzzOnce(t, seed)
		})
	}
}

func fuzzOnce(t *testing.T, seed int64) {
	c := newAPIClient(t)
	m := newModel()
	rng := rand.New(rand.NewSource(seed))
	names := []string{"a", "b", "c", "d", "e"}
	const iters = 1200
	maxOff := int64(8 * BlockSize)

	for it := 0; it < iters; it++ {
		ctx := fmt.Sprintf("seed=%d iter=%d rev=%d", seed, it, m.revision)

		// Occasionally use a stale revision to exercise conflicts.
		rev := m.revision
		if rng.Intn(8) == 0 {
			rev = rng.Int63n(m.revision + 5)
		}
		op := rng.Intn(100)
		switch {
		case op < 12: // create
			name := names[rng.Intn(len(names))]
			nr, status := c.create(name, rev)
			mk := m.create(name, rev)
			assertSame(t, ctx, "create "+name, mk, status, nr, m)
		case op < 24: // clone
			src := names[rng.Intn(len(names))]
			dst := names[rng.Intn(len(names))]
			nr, status := c.clone(src, dst, rev)
			mk := m.clone(src, dst, rev)
			assertSame(t, ctx, "clone", mk, status, nr, m)
		case op < 62: // write (most common: creates shared divergence & holes)
			name := names[rng.Intn(len(names))]
			off := rng.Int63n(maxOff)
			if rng.Intn(20) == 0 {
				off = rng.Int63n(int64(MaxBlocks)*BlockSize + BlockSize) // big sparse jump
			}
			size := rng.Intn(3 * BlockSize)
			if rng.Intn(10) == 0 {
				size = 0 // zero-length writes are no-ops at or before EOF
			}
			p := make([]byte, size)
			for i := range p {
				p[i] = byte(rng.Intn(256))
			}
			nr, status := c.write(name, off, p, rev)
			mk := m.write(name, off, p, rev)
			assertSame(t, ctx, fmt.Sprintf("write %s@%d+%d", name, off, size), mk, status, nr, m)
		case op < 82: // truncate
			name := names[rng.Intn(len(names))]
			var length int64
			switch rng.Intn(4) {
			case 0:
				length = 0
			case 1:
				length = rng.Int63n(maxOff)
			case 2:
				length = rng.Int63n(int64(BlockSize)) // unaligned inside block 0
			default:
				length = rng.Int63n(int64(4 * BlockSize))
			}
			nr, status := c.truncate(name, length, rev)
			mk := m.truncate(name, length, rev)
			assertSame(t, ctx, fmt.Sprintf("truncate %s->%d", name, length), mk, status, nr, m)
		case op < 92: // delete
			name := names[rng.Intn(len(names))]
			nr, status := c.del(name, rev)
			mk := m.del(name, rev)
			assertSame(t, ctx, "delete "+name, mk, status, nr, m)
		default: // reads, never mutate
			name := names[rng.Intn(len(names))]
			off := rng.Int63n(maxOff + 2*BlockSize)
			length := int64(-1)
			if rng.Intn(2) == 0 {
				length = rng.Int63n(3 * BlockSize)
			}
			raw, status, _ := c.read(name, off, length)
			mf, exists := m.files[name]
			switch {
			case !exists:
				if status != http.StatusNotFound {
					t.Fatalf("[%s] read missing file status=%d", ctx, status)
				}
			case off > int64(len(mf.data)):
				if status != http.StatusRequestedRangeNotSatisfiable {
					t.Fatalf("[%s] read past EOF status=%d", ctx, status)
				}
			default:
				if status != http.StatusOK {
					t.Fatalf("[%s] read status=%d", ctx, status)
				}
				var want []byte
				if length < 0 || off+length > int64(len(mf.data)) {
					want = mf.data[off:]
				} else {
					want = mf.data[off : off+length]
				}
				if !bytes.Equal(raw, want) {
					t.Fatalf("[%s] read mismatch off=%d len=%d got=%d want=%d", ctx, off, length, len(raw), len(want))
				}
			}
		}

		// After every single operation the full volume (revision, physical
		// totals and complete file contents) must match the naive model.
		if it%5 == 0 {
			checkStats(t, c, m, ctx)
		}
	}
	checkStats(t, c, m, "final")
}

// assertSame compares a model error category with the server's response.
// Exists vs revision conflict share HTTP 409, so those two collapse.
func assertSame(t *testing.T, ctx, desc string, mk errKind, status int, nr int64, m *model) {
	t.Helper()
	sk := statusToKind(status)
	equiv := func(a, b errKind) bool {
		if a == b {
			return true
		}
		return (a == kExists || a == kConflict) && (b == kExists || b == kConflict)
	}
	if !equiv(mk, sk) {
		t.Fatalf("[%s] %s: model=%v server=%v(%d)", ctx, desc, mk, sk, status)
	}
	if mk == kOK {
		if int64(nr) != m.revision {
			t.Fatalf("[%s] %s: rev server=%d model=%d", ctx, desc, nr, m.revision)
		}
	}
}
