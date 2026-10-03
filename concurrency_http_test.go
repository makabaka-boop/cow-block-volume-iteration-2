package vfs

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// TestHTTPConcurrentSameRevision fires many POSTs carrying the same expected
// revision against the real HTTP stack. Exactly one must return 200; the rest
// must be revision conflicts with no state corruption.
func TestHTTPConcurrentSameRevision(t *testing.T) {
	srv := httptest.NewServer(NewServer(New()).Handler())
	defer srv.Close()

	post := func(path string, body []byte) (int, int64) {
		resp, err := srv.Client().Post(srv.URL+path, "application/octet-stream", bytes.NewReader(body))
		if err != nil {
			t.Error(err)
			return 0, -1
		}
		defer resp.Body.Close()
		var mr struct {
			Revision int64 `json:"revision"`
		}
		_ = json.NewDecoder(resp.Body).Decode(&mr)
		return resp.StatusCode, mr.Revision
	}

	if code, _ := post("/create?name=f&rev=0", nil); code != http.StatusOK {
		t.Fatalf("create code=%d", code)
	}

	const n = 100
	var wg sync.WaitGroup
	codes := make([]int, n)
	revs := make([]int64, n)
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			// Every goroutine attempts the same one-byte write at the same
			// revision: exactly one may commit.
			codes[i], revs[i] = post("/write?name=f&offset=0&rev=1", []byte{0x5A})
		}(i)
	}
	wg.Wait()

	ok := 0
	for i, code := range codes {
		switch code {
		case http.StatusOK:
			ok++
			if revs[i] != 2 {
				t.Fatalf("winner got rev %d, want 2", revs[i])
			}
		case http.StatusConflict:
			// expected for losers
		default:
			t.Fatalf("unexpected code %d", code)
		}
	}
	if ok != 1 {
		t.Fatalf("exactly one writer must win, got %d/%d", ok, n)
	}

	// The winning one-byte write leaves the file at revision 2 with one block.
	resp, err := srv.Client().Get(srv.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var st Stats
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		t.Fatal(err)
	}
	if st.Revision != 2 {
		t.Fatalf("revision=%d want 2", st.Revision)
	}
	if st.LogicalFiles != 1 || st.UsedBlocks != 1 {
		t.Fatalf("unexpected stats: %+v", st)
	}
}

// TestHTTPRevisionHeader checks X-Expected-Revision as an alternative to rev=.
func TestHTTPRevisionHeader(t *testing.T) {
	srv := httptest.NewServer(NewServer(New()).Handler())
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/create?name=h", nil)
	req.Header.Set("X-Expected-Revision", "0")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}

	req, _ = http.NewRequest(http.MethodPost, srv.URL+"/truncate?name=h&length=10", nil)
	req.Header.Set("X-Expected-Revision", "1")
	resp, err = srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}

	// Missing revision is a 400.
	resp, err = srv.Client().Post(srv.URL+"/delete?name=h", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status=%d want 400", resp.StatusCode)
	}
}
