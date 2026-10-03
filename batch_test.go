package vfs

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"sync"
	"testing"
)

// snapshot returns a complete independent copy used by the batch oracle.
func (m *model) snapshot() *model {
	cp := &model{
		files:    make(map[string]*modelFile, len(m.files)),
		blocks:   make(map[int]*modelBlock, len(m.blocks)),
		revision: m.revision,
		nextID:   m.nextID,
	}
	for id, b := range m.blocks {
		cp.blocks[id] = &modelBlock{id: id, refs: b.refs}
	}
	for name, f := range m.files {
		nf := &modelFile{
			data: append([]byte(nil), f.data...),
			phys: make(map[int64]int, len(f.phys)),
		}
		for idx, id := range f.phys {
			nf.phys[idx] = id
		}
		cp.files[name] = nf
	}
	return cp
}

func (m *model) batch(steps []BatchStep, rev int64) (newRev int64, kind errKind, step int) {
	if len(steps) == 0 || len(steps) > MaxBatchSteps {
		return m.revision, kBadName, 0 // both are request-shape errors over HTTP
	}
	if rev != m.revision {
		return m.revision, kConflict, 0
	}
	stage := m.snapshot()
	for i, s := range steps {
		switch s.Op {
		case "create":
			kind = stage.create(s.Name, stage.revision)
		case "clone":
			kind = stage.clone(s.Src, s.Dst, stage.revision)
		case "write":
			kind = stage.write(s.Name, s.Offset, s.Data, stage.revision)
		case "truncate":
			kind = stage.truncate(s.Name, s.Length, stage.revision)
		case "delete":
			kind = stage.del(s.Name, stage.revision)
		default:
			kind = kBadName
		}
		if kind != kOK {
			return m.revision, kind, i + 1
		}
	}
	// A batch is one commit even when an individual step is a non-mutating
	// zero-length write.
	stage.revision = m.revision + 1
	*m = *stage
	return stage.revision, kOK, 0
}

func (c *apiClient) batch(steps []BatchStep, rev int64) (newRev int64, status int, step int, body string) {
	raw, err := json.Marshal(batchRequest{Steps: steps})
	if err != nil {
		panic(err)
	}
	resp, err := c.hc.Post(c.srv.URL+"/batch?rev="+itoa(rev), "application/json", bytes.NewReader(raw))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	body = string(data)
	if resp.StatusCode == http.StatusOK {
		var mr mutResp
		if err := json.Unmarshal(data, &mr); err != nil {
			panic(err)
		}
		return mr.Revision, resp.StatusCode, 0, body
	}
	var er errorResponse
	_ = json.Unmarshal(data, &er)
	return -1, resp.StatusCode, er.Step, body
}

func itoa(n int64) string { return fmt.Sprintf("%d", n) }

func TestBatchSharedForkAndDeleteCreate(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()

	check := func(ctx string, steps []BatchStep, wantRev int64) {
		t.Helper()
		nrS, status, stepS, _ := c.batch(steps, m.revision)
		nrM, kind, _ := m.batch(steps, m.revision)
		if status != http.StatusOK || kind != kOK {
			t.Fatalf("%s: server status=%d, model=%v (steps=%d)", ctx, status, kind, stepS)
		}
		if nrS != nrM || nrS != wantRev {
			t.Fatalf("%s: rev server=%d model=%d want=%d", ctx, nrS, nrM, wantRev)
		}
		checkStats(t, c, m, ctx)
	}

	check("create/write/clone", []BatchStep{
		{Op: "create", Name: "a"},
		{Op: "write", Name: "a", Data: bytesFill(BlockSize, 0x11)},
		{Op: "clone", Src: "a", Dst: "b"},
	}, 1)

	// The first staged step forks a block shared before the batch. The clone
	// in the second step then shares that newly private fork, and the final
	// write forks it once more for the new clone.
	check("staged forks", []BatchStep{
		{Op: "write", Name: "b", Data: []byte("FORK")},
		{Op: "clone", Src: "b", Dst: "c"},
		{Op: "write", Name: "c", Offset: 5, Data: []byte("X")},
	}, 2)

	a, status := c.read2("a", 0, 4)
	if status != http.StatusOK || !bytes.Equal(a, bytesFill(4, 0x11)) {
		t.Fatalf("original changed: % x status=%d", a, status)
	}
	b, status := c.read2("b", 0, 4)
	if status != http.StatusOK || !bytes.Equal(b, []byte("FORK")) {
		t.Fatalf("first fork changed: % x status=%d", b, status)
	}
	cBytes, status := c.read2("c", 0, 6)
	wantC := append(bytesFill(5, 0x11), 'X')
	copy(wantC, []byte("FORK"))
	if status != http.StatusOK || !bytes.Equal(cBytes, wantC) {
		t.Fatalf("second fork changed: % x status=%d", cBytes, status)
	}

	// Delete a name, recreate it and write in the same transaction. Freed
	// physical IDs are available to later steps.
	check("delete then create", []BatchStep{
		{Op: "delete", Name: "a"},
		{Op: "create", Name: "a"},
		{Op: "write", Name: "a", Data: []byte("new-data")},
	}, 3)
	a, status = c.read2("a", 0, -1)
	if status != http.StatusOK || !bytes.Equal(a, []byte("new-data")) {
		t.Fatalf("recreated file content=%q status=%d", a, status)
	}
}

func TestBatchCriticalQuotaAndMidFailure(t *testing.T) {
	c := newAPIClient(t)
	m := newModel()
	twoBlocks := bytesFill(2*BlockSize, 0xAA)
	setup := []BatchStep{
		{Op: "create", Name: "f"},
		{Op: "write", Name: "f", Data: twoBlocks},
		{Op: "clone", Src: "f", Dst: "g"},
		{Op: "create", Name: "h"},
		{Op: "write", Name: "h", Data: bytesFill((MaxBlocks-2)*BlockSize, 0xBB)},
	}
	nrS, status, _, _ := c.batch(setup, 0)
	nrM, kind, _ := m.batch(setup, 0)
	if status != http.StatusOK || kind != kOK || nrS != 1 || nrM != 1 {
		t.Fatalf("setup status=%d model=%v rev=%d/%d", status, kind, nrS, nrM)
	}
	checkStats(t, c, m, "full volume")

	failing := []BatchStep{
		{Op: "delete", Name: "f"}, // block remains alive through clone g
		{Op: "delete", Name: "g"}, // now two blocks enter the staged free pool
		{Op: "create", Name: "x"},
		{Op: "write", Name: "x", Data: bytesFill(3*BlockSize, 0xCC)},
	}
	nrS, status, step, body := c.batch(failing, 1)
	nrM, kind, modelStep := m.batch(failing, 1)
	if status != http.StatusInsufficientStorage || kind != kQuota || step != 4 || modelStep != 4 {
		t.Fatalf("failure status=%d step=%d body=%q model=%v/%d", status, step, body, kind, modelStep)
	}
	if nrS != -1 || nrM != 1 {
		t.Fatalf("failed batch revisions server marker=%d model=%d", nrS, nrM)
	}
	checkStats(t, c, m, "after rolled-back batch")
	if st := c.statsForTest(); st.Revision != 1 || st.UsedBlocks != MaxBlocks || st.FreeBlocks != 0 {
		t.Fatalf("real volume/free pool changed: %+v", st)
	}

	// Two blocks were reclaimed by the staged deletes; a later write using
	// exactly two blocks must therefore succeed at the critical quota edge.
	critical := failing[:3]
	critical = append(critical, BatchStep{Op: "write", Name: "x", Data: bytesFill(2*BlockSize, 0xDD)})
	nrS, status, step, body = c.batch(critical, 1)
	nrM, kind, modelStep = m.batch(critical, 1)
	if status != http.StatusOK || kind != kOK || step != 0 || modelStep != 0 {
		t.Fatalf("critical batch status=%d step=%d body=%q model=%v/%d", status, step, body, kind, modelStep)
	}
	if nrS != 2 || nrM != 2 {
		t.Fatalf("critical revision server=%d model=%d", nrS, nrM)
	}
	checkStats(t, c, m, "critical quota committed")
}

func (c *apiClient) statsForTest() Stats {
	resp, err := c.hc.Get(c.srv.URL + "/stats")
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	var st Stats
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		panic(err)
	}
	return st
}

func TestHTTPBatchConcurrentSameRevision(t *testing.T) {
	srv := newAPIClient(t)
	if nr, status, _, _ := srv.batch([]BatchStep{{Op: "create", Name: "f"}}, 0); status != http.StatusOK || nr != 1 {
		t.Fatalf("initial batch status=%d rev=%d", status, nr)
	}

	const n = 100
	var wg sync.WaitGroup
	codes := make([]int, n)
	revs := make([]int64, n)
	steps := []BatchStep{{Op: "write", Name: "f", Data: []byte{0x5A}}}
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			revs[i], codes[i], _, _ = srv.batch(steps, 1)
		}(i)
	}
	wg.Wait()

	ok := 0
	for i, code := range codes {
		switch code {
		case http.StatusOK:
			ok++
			if revs[i] != 2 {
				t.Fatalf("winner rev=%d", revs[i])
			}
		case http.StatusConflict:
		default:
			t.Fatalf("unexpected status=%d", code)
		}
	}
	if ok != 1 {
		t.Fatalf("winners=%d, want 1", ok)
	}
	st := srv.statsForTest()
	if st.Revision != 2 || st.LogicalFiles != 1 || st.UsedBlocks != 1 {
		t.Fatalf("unexpected final stats: %+v", st)
	}
}

func TestBatchValidationAndLocalFailure(t *testing.T) {
	v := New()
	if _, err := v.Create("a", 0); err != nil {
		t.Fatal(err)
	}
	for _, steps := range [][]BatchStep{nil, {}} {
		if _, err := v.Batch(steps, 1); !errors.Is(err, ErrInvalidBatch) {
			t.Fatalf("empty batch err=%v", err)
		}
	}
	many := make([]BatchStep, MaxBatchSteps+1)
	for i := range many {
		many[i] = BatchStep{Op: "write", Name: "a", Data: []byte{1}}
	}
	if _, err := v.Batch(many, 1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("long batch err=%v", err)
	}

	_, err := v.Batch([]BatchStep{
		{Op: "create", Name: "b"},
		{Op: "write", Name: "missing", Data: []byte("x")},
	}, 1)
	var be *BatchError
	if !errors.As(err, &be) || be.Step != 2 || !errors.Is(be, ErrNotFound) {
		t.Fatalf("want step 2 not found, got %v", err)
	}
	if v.Revision() != 1 {
		t.Fatalf("failed batch changed revision to %d", v.Revision())
	}
	if _, err := v.Read("b", 0, 1); !errors.Is(err, ErrNotFound) {
		t.Fatalf("earlier staged step leaked: %v", err)
	}
}

func TestBatchModelFuzzHTTP(t *testing.T) {
	for _, seed := range []int64{11, 22, 33} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			c := newAPIClient(t)
			m := newModel()
			rng := rand.New(rand.NewSource(seed))
			names := []string{"a", "b", "c", "d"}

			for it := 0; it < 250; it++ {
				steps := make([]BatchStep, 1+rng.Intn(MaxBatchSteps))
				for j := range steps {
					name := names[rng.Intn(len(names))]
					switch rng.Intn(100) {
					case 0:
						steps[j] = BatchStep{Op: "create", Name: name}
					case 1:
						steps[j] = BatchStep{Op: "clone", Src: names[rng.Intn(len(names))], Dst: name}
					case 2:
						steps[j] = BatchStep{Op: "delete", Name: name}
					case 3:
						steps[j] = BatchStep{Op: "truncate", Name: name, Length: rng.Int63n(6 * BlockSize)}
					default:
						size := rng.Intn(3 * BlockSize)
						if rng.Intn(12) == 0 {
							size = 0
						}
						data := make([]byte, size)
						for i := range data {
							data[i] = byte(rng.Intn(256))
						}
						off := rng.Int63n(8 * BlockSize)
						if rng.Intn(20) == 0 {
							off = rng.Int63n(int64(MaxBlocks)*BlockSize + BlockSize)
						}
						steps[j] = BatchStep{Op: "write", Name: name, Offset: off, Data: data}
					}
				}

				rev := m.revision
				if rng.Intn(6) == 0 && rev > 0 {
					rev--
				}
				nrS, status, stepS, body := c.batch(steps, rev)
				nrM, kind, stepM := m.batch(steps, rev)
				equiv := (kind == kExists || kind == kConflict) &&
					(status == http.StatusConflict)
				if !equiv && statusToKind(status) != kind {
					t.Fatalf("seed=%d iter=%d status=%d model=%v body=%q", seed, it, status, kind, body)
				}
				if kind != kOK && stepS != stepM {
					t.Fatalf("seed=%d iter=%d error step server=%d model=%d", seed, it, stepS, stepM)
				}
				if kind == kOK && nrS != nrM {
					t.Fatalf("seed=%d iter=%d rev server=%d model=%d", seed, it, nrS, nrM)
				}
				if it%10 == 0 || kind == kOK {
					checkStats(t, c, m, fmt.Sprintf("seed=%d iter=%d", seed, it))
				}
			}
			checkStats(t, c, m, "final")
		})
	}
}
