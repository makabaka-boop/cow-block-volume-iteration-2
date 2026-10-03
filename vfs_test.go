package vfs

import (
	"bytes"
	"errors"
	"sync"
	"testing"
)

func bytesFill(n int, c byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = c
	}
	return b
}

// TestCrossBlockWrite writes data straddling the 4096 boundary at several
// offsets and checks the surrounding bytes (holes on both sides) read back
// exactly.
func TestCrossBlockWrite(t *testing.T) {
	v := New()
	if rev, err := v.Create("f", 0); err != nil || rev != 1 {
		t.Fatalf("create: rev=%d err=%v", rev, err)
	}

	cases := []struct {
		offset int64
		size   int
		fill   byte
	}{
		{4090, 20, 0xAA}, // straddles block 0/1
		{8180, 30, 0xBB}, // straddles block 1/2, overlaps previous tail in block1
		{12287, 4, 0xCC}, // end exactly at a boundary
		{16384, 1, 0xDD}, // isolated byte in a fresh block
	}
	rev := int64(1)
	want := make(map[int64]byte)
	for _, c := range cases {
		data := bytesFill(c.size, c.fill)
		nr, err := v.Write("f", c.offset, data, rev)
		if err != nil {
			t.Fatalf("write@%d: %v", c.offset, err)
		}
		rev = nr
		for i := 0; i < c.size; i++ {
			want[c.offset+int64(i)] = c.fill
		}
	}

	length := int64(16385)
	got, err := v.Read("f", 0, -1)
	if err != nil || int64(len(got)) != length {
		t.Fatalf("read: len=%d err=%v", len(got), err)
	}
	for i := int64(0); i < length; i++ {
		exp := byte(0)
		if b, ok := want[i]; ok {
			exp = b
		}
		if got[i] != exp {
			t.Fatalf("byte %d: got %#x want %#x", i, got[i], exp)
		}
	}
	st := v.Stats()
	if st.Files[0].Length != length {
		t.Fatalf("length = %d, want %d", st.Files[0].Length, length)
	}
	// Writes touch blocks 0,1,2,3 and 4; no holes among them.
	if pb := st.Files[0].PhysicalBlocks; pb != 5 {
		t.Fatalf("physical blocks = %d, want 5", pb)
	}
}

// TestSparseHoles verifies unwritten regions read as zeros and consume no
// physical blocks, including a write far beyond EOF.
func TestSparseHoles(t *testing.T) {
	v := New()
	rev, _ := v.Create("s", 0)

	// One byte at 10*BS+100 => length is that +1; exactly one physical block.
	writeAt := int64(10*BlockSize + 100)
	nr, err := v.Write("s", writeAt, []byte{0x7F}, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	wantLen := writeAt + 1
	out, err := v.Read("s", 0, -1)
	if err != nil || int64(len(out)) != wantLen {
		t.Fatalf("read len=%d err=%v", len(out), err)
	}
	if out[0] != 0 || out[writeAt-1] != 0 || out[writeAt] != 0x7F {
		t.Fatal("sparse content mismatch")
	}
	if id := v.DebugBlockID("s", 0); id != -1 {
		t.Fatalf("block 0 should be a hole, got id %d", id)
	}
	if id := v.DebugBlockID("s", 10); id < 0 {
		t.Fatal("block 10 should be mapped")
	}
	st := v.Stats()
	if st.UsedBlocks != 1 {
		t.Fatalf("used = %d, want 1", st.UsedBlocks)
	}
	fi := st.Files[0]
	if fi.Length != wantLen || fi.PhysicalBlocks != 1 {
		t.Fatalf("file stats wrong: %+v", fi)
	}
	// Logical bytes with no backing block: the 10 full hole blocks. The 100
	// bytes before the written byte live in the same materialized block and
	// are therefore not storage holes even though they read as zero.
	wantSparse := int64(10 * BlockSize)
	wantMapped := wantLen - wantSparse // only the in-file part of block 10
	if st.SparseBytes != wantSparse {
		t.Fatalf("sparse bytes = %d, want %d", st.SparseBytes, wantSparse)
	}
	if st.SparseBytes+wantMapped != wantLen {
		t.Fatalf("sparse+mapped = %d, want length %d", st.SparseBytes+wantMapped, wantLen)
	}
	if _, err := v.Write("s", wantLen, nil, rev); err != nil {
		t.Fatalf("zero write at EOF: %v", err)
	}
	if after := v.Stats(); after.SparseBytes != wantSparse || after.Files[0].Length != wantLen || after.Revision != rev {
		t.Fatalf("zero write changed stats: %+v (want sparse=%d, length=%d, rev=%d)", after, wantSparse, wantLen, rev)
	}
	if _, err := v.Write("s", wantLen+1, nil, rev); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("zero write past EOF: %v", err)
	}
	_ = rev
}

// TestCloneFork checks that a clone shares physical blocks at first, diverges
// only after a write to a shared block, and that neither side can corrupt the
// other.
func TestCloneFork(t *testing.T) {
	v := New()
	rev, _ := v.Create("a", 0)
	data := bytesFill(2*BlockSize, 0xAB)
	nr, err := v.Write("a", 0, data, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr

	b0 := v.DebugBlockID("a", 0)
	b1 := v.DebugBlockID("a", 1)
	if b0 < 0 || b1 < 0 || b0 == b1 {
		t.Fatal("expected two distinct blocks")
	}
	if st := v.Stats(); st.UsedBlocks != 2 {
		t.Fatalf("used before clone = %d", st.UsedBlocks)
	}

	nr, err = v.Clone("a", "b", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	// Blocks are shared after clone: still 2 physical blocks.
	if st := v.Stats(); st.UsedBlocks != 2 {
		t.Fatalf("used after clone = %d, want 2", st.UsedBlocks)
	}
	if v.DebugBlockID("b", 0) != b0 || v.DebugBlockID("b", 1) != b1 {
		t.Fatal("clone must share physical blocks")
	}

	// First write to b's block 0 triggers a copy; block 1 remains shared.
	nr, err = v.Write("b", 10, []byte{0x01, 0x02, 0x03}, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 3 {
		t.Fatalf("used after COW = %d, want 3", st.UsedBlocks)
	}
	if v.DebugBlockID("b", 0) == b0 {
		t.Fatal("b block 0 should have been copied")
	}
	if v.DebugBlockID("b", 1) != b1 {
		t.Fatal("b block 1 should still be shared")
	}
	a, _ := v.Read("a", 0, 20)
	b, _ := v.Read("b", 0, 20)
	expA := bytesFill(20, 0xAB)
	if !bytes.Equal(a, expA) {
		t.Fatal("a was corrupted by b's write")
	}
	if !bytes.Equal(b[:10], bytesFill(10, 0xAB)) || !bytes.Equal(b[10:13], []byte{1, 2, 3}) || !bytes.Equal(b[13:], bytesFill(7, 0xAB)) {
		t.Fatalf("b content wrong: % x", b)
	}
}

// TestTruncateRelease covers shrink, growth (holes), tail zeroing of a shared
// straddling block, and full deletion releasing references.
func TestTruncateRelease(t *testing.T) {
	v := New()
	rev, _ := v.Create("t", 0)
	nr, err := v.Write("t", 0, bytesFill(3*BlockSize, 0x55), rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	nr, err = v.Clone("t", "c", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 3 {
		t.Fatalf("used = %d", st.UsedBlocks)
	}

	// Shrink t to 1.5 blocks: drop block 2 (still shared with c => stays),
	// zero tail of block 1 (shared => COW). Used goes 3 -> 4.
	nr, err = v.Truncate("t", BlockSize+10, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 4 {
		t.Fatalf("used after partial truncate = %d, want 4", st.UsedBlocks)
	}
	tail, _ := v.Read("t", BlockSize, -1)
	if len(tail) != 10 || !bytes.Equal(tail, bytesFill(10, 0x55)) {
		t.Fatalf("tail wrong: % x", tail)
	}
	// The clone must be untouched, block 1 still 0x55 to its end.
	cb, _ := v.Read("c", BlockSize+10, 20)
	if !bytes.Equal(cb, bytesFill(20, 0x55)) {
		t.Fatal("clone corrupted by truncate zeroing")
	}

	// Truncate t to 0 frees its private blocks; shared ones were already gone.
	nr, err = v.Truncate("t", 0, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	// t holds 0 blocks; c holds the original 3 => used == 3.
	if st := v.Stats(); st.UsedBlocks != 3 {
		t.Fatalf("used after zero truncate = %d, want 3", st.UsedBlocks)
	}

	// Grow via truncate creates a hole: no allocation.
	nr, err = v.Truncate("t", 5*BlockSize, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 3 {
		t.Fatalf("grow allocated blocks: used = %d", st.UsedBlocks)
	}
	z, _ := v.Read("t", 4*BlockSize, BlockSize)
	if !bytes.Equal(z, make([]byte, BlockSize)) {
		t.Fatal("grown region not zero")
	}

	// Deleting c frees the last three blocks.
	nr, err = v.Delete("c", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != 0 {
		t.Fatalf("used after delete = %d, want 0", st.UsedBlocks)
	}
	// Deleting again is a conflict-free 404 at the current revision.
	if _, err := v.Delete("c", rev); !errors.Is(err, ErrNotFound) {
		t.Fatalf("want ErrNotFound, got %v", err)
	}
}

// TestQuotaAtomic fills nearly the whole volume, then attempts a write that
// cannot be reserved: it must fail with ErrQuota and leave no partial data
// and no dangling references.
func TestQuotaAtomic(t *testing.T) {
	v := New()
	rev, _ := v.Create("q", 0)
	// Occupy MaxBlocks-2 blocks with one file.
	big := bytesFill((MaxBlocks-2)*BlockSize, 0x11)
	nr, err := v.Write("q", 0, big, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != MaxBlocks-2 {
		t.Fatalf("used = %d", st.UsedBlocks)
	}
	nr, err = v.Create("q2", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr

	// A write spanning 4 fresh blocks needs 4 allocations but only 2 exist.
	// No block is shared, so the whole thing must abort untouched.
	before := v.Stats()
	_, err = v.Write("q2", BlockSize/2, bytesFill(4*BlockSize, 0x22), rev)
	if !errors.Is(err, ErrQuota) {
		t.Fatalf("want ErrQuota, got %v", err)
	}
	after := v.Stats()
	if after.UsedBlocks != before.UsedBlocks || after.Revision != before.Revision {
		t.Fatal("failed write changed volume state")
	}
	if _, err := v.Read("q2", 0, 1); err != nil {
		t.Fatalf("q2 should still be empty: %v", err)
	}

	// Exactly 2 fresh blocks fits: succeeds.
	nr, err = v.Write("q2", 0, bytesFill(2*BlockSize, 0x33), rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != MaxBlocks {
		t.Fatalf("used = %d, want %d", st.UsedBlocks, MaxBlocks)
	}
	// A one-byte write into a new block now fails with no mutation.
	_, err = v.Write("q2", 10*BlockSize, []byte{1}, rev)
	if !errors.Is(err, ErrQuota) {
		t.Fatalf("want ErrQuota, got %v", err)
	}
}

// TestTruncateQuotaAtomic: shrinking a shared straddling block that needs COW
// while the volume is completely full must fail atomically.
func TestTruncateQuotaAtomic(t *testing.T) {
	v := New()
	rev, _ := v.Create("x", 0)
	nr, err := v.Write("x", 0, bytesFill(BlockSize, 0x44), rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	// Fill the rest of the volume with a second file's blocks.
	nr, err = v.Create("y", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	nr, err = v.Write("y", 0, bytesFill((MaxBlocks-1)*BlockSize, 0x66), rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	// Clone x: shares its single block; volume remains exactly full.
	nr, err = v.Clone("x", "z", rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	// Shrinking z to an unaligned length needs one private copy: no room,
	// even though nothing is being dropped yet.
	_, err = v.Truncate("z", 10, rev)
	if !errors.Is(err, ErrQuota) {
		t.Fatalf("want ErrQuota, got %v", err)
	}
	// But shrinking below a block boundary (no straddle copy) succeeds and
	// releases nothing physical (block still shared with x).
	nr, err = v.Truncate("z", 0, rev)
	if err != nil {
		t.Fatalf("truncate to zero: %v", err)
	}
	rev = nr
	if st := v.Stats(); st.UsedBlocks != MaxBlocks {
		t.Fatalf("used = %d, want %d", st.UsedBlocks, MaxBlocks)
	}
}

// TestRevisionConflict drives every mutation kind through a stale revision.
func TestRevisionConflict(t *testing.T) {
	v := New()
	if _, err := v.Create("r", 0); err != nil {
		t.Fatal(err)
	}
	if _, err := v.Create("r2", 0); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale create: %v", err)
	}
	if _, err := v.Write("r", 0, []byte{1}, 0); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale write: %v", err)
	}
	if _, err := v.Truncate("r", 1, 0); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale truncate: %v", err)
	}
	if _, err := v.Clone("r", "rr", 0); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale clone: %v", err)
	}
	if _, err := v.Delete("r", 0); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale delete: %v", err)
	}
	if v.Revision() != 1 {
		t.Fatal("failed mutations must not bump revision")
	}

	// Truncate is still an explicit mutation when the size is unchanged.
	nr, err := v.Truncate("r", 0, 1)
	if err != nil || nr != 2 {
		t.Fatalf("same-size truncate: rev=%d err=%v", nr, err)
	}
	// A zero-length write is not a mutation and must not grow the file or
	// advance the revision.
	nr, err = v.Write("r", 0, nil, 2)
	if err != nil || nr != 2 {
		t.Fatalf("zero write: rev=%d err=%v", nr, err)
	}
	if _, err := v.Write("r", 100, nil, 2); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("zero write past EOF: %v", err)
	}
	if v.Revision() != 2 {
		t.Fatal("zero-length write changed revision")
	}
}

// TestConcurrentSameRevision fires many identical mutations concurrently: at
// most one may commit, and volume invariants must hold afterwards.
func TestConcurrentSameRevision(t *testing.T) {
	v := New()
	if _, err := v.Create("w", 0); err != nil {
		t.Fatal(err)
	}

	const n = 64
	var wg sync.WaitGroup
	errs := make([]error, n)
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			// Every goroutine attempts the same write at the same revision:
			// exactly one may commit, leaving a single byte in a single block.
			_, errs[i] = v.Write("w", 0, []byte{0x5A}, 1)
		}(i)
	}
	wg.Wait()
	ok, conflicts := 0, 0
	for _, err := range errs {
		switch {
		case err == nil:
			ok++
		case errors.Is(err, ErrConflict):
			conflicts++
		default:
			t.Fatalf("unexpected err %v", err)
		}
	}
	if ok != 1 || conflicts != n-1 {
		t.Fatalf("winners=%d conflicts=%d", ok, conflicts)
	}
	if v.Revision() != 2 {
		t.Fatalf("revision = %d, want 2", v.Revision())
	}
	data, _ := v.Read("w", 0, -1)
	if len(data) != 1 || data[0] != 0x5A {
		t.Fatalf("content = % x", data)
	}
	if st := v.Stats(); st.UsedBlocks != 1 {
		t.Fatalf("used blocks = %d, want 1", st.UsedBlocks)
	}
}

// TestConcurrentMixed hammers the volume with chained revisions from many
// goroutines; the counter must end exactly at the number of successes.
func TestConcurrentMixed(t *testing.T) {
	v := New()
	if _, err := v.Create("m", 0); err != nil {
		t.Fatal(err)
	}
	const n = 200
	var wg sync.WaitGroup
	var mu sync.Mutex
	successes := 0
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			for attempt := 0; attempt < 50; attempt++ {
				rev := v.Revision()
				_, err := v.Write("m", int64(i%32), []byte{byte(i + 1)}, rev)
				if err == nil {
					mu.Lock()
					successes++
					mu.Unlock()
					return
				}
				if !errors.Is(err, ErrConflict) {
					t.Errorf("unexpected err: %v", err)
					return
				}
			}
		}(i)
	}
	wg.Wait()
	if v.Revision() != int64(successes)+1 {
		t.Fatalf("rev=%d successes=%d", v.Revision(), successes)
	}
	data, _ := v.Read("m", 0, -1)
	if len(data) != 32 {
		t.Fatalf("len = %d, want 32", len(data))
	}
}

// TestReadRanges exercises boundary reads and error cases.
func TestReadRanges(t *testing.T) {
	v := New()
	rev, _ := v.Create("rd", 0)
	nr, err := v.Write("rd", BlockSize-2, []byte{1, 2, 3, 4}, rev)
	if err != nil {
		t.Fatal(err)
	}
	rev = nr
	if b, err := v.Read("rd", BlockSize-2, 4); err != nil || !bytes.Equal(b, []byte{1, 2, 3, 4}) {
		t.Fatalf("exact range: % x, %v", b, err)
	}
	if b, err := v.Read("rd", BlockSize+1, -1); err != nil || !bytes.Equal(b, []byte{4}) {
		t.Fatalf("to-EOF range: % x, %v", b, err)
	}
	if b, err := v.Read("rd", BlockSize+2, -1); err != nil || len(b) != 0 {
		t.Fatalf("at EOF: % x, %v", b, err)
	}
	if _, err := v.Read("rd", BlockSize+3, -1); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("past EOF: %v", err)
	}
	if _, err := v.Read("rd", -1, 1); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("negative offset: %v", err)
	}
	if _, _, err := v.ReadWithRevision("rd", 1, int64Max); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("overflowing range: %v", err)
	}
	if _, _, err := v.ReadWithRevision("rd", int64Max, 1); !errors.Is(err, ErrOutOfRange) {
		t.Fatalf("overflowing end: %v", err)
	}
	_ = nr
}

// TestNames enforces the unique ASCII name policy.
func TestNames(t *testing.T) {
	good := []string{"a", "A", "1name", "f1.dat", "x_y-z.1", "abcDEF123", "a.."}
	bad := []string{"", ".hidden", "_x", "-y", "a/b", "a b", "中文", "a\x00b", "a:b"}
	for _, n := range good {
		if !ValidateName(n) {
			t.Errorf("expected valid: %q", n)
		}
	}
	for _, n := range bad {
		if ValidateName(n) {
			t.Errorf("expected invalid: %q", n)
		}
	}
}
