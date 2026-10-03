package vfs

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// maxWriteBytes caps a single overwrite request body.
const maxWriteBytes = 64 << 20 // 64 MiB

// maxBatchBytes caps an atomic batch request body (JSON, base64 write data).
const maxBatchBytes = maxWriteBytes + 4<<20

// Server exposes a Volume over HTTP.
type Server struct {
	Vol *Volume
}

// NewServer wraps vol.
func NewServer(vol *Volume) *Server {
	if vol == nil {
		vol = New()
	}
	return &Server{Vol: vol}
}

// Handler returns the root http.Handler.
//
// Mutations (all POST) take the expected revision via the "rev" query
// parameter or the X-Expected-Revision header and return
// {"revision": <new revision>}.
//
//	POST /create?name=...
//	POST /clone?src=...&dst=...
//	POST /write?name=...&offset=N        body: bytes to overwrite
//	POST /truncate?name=...&length=N
//	POST /delete?name=...
//	POST /batch                          body: {"revision": N, "steps": [...]}
//	GET  /read?name=...&offset=N[&length=N]   (no length => to EOF)
//	GET  /stat?name=...
//	GET  /stats
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/create", s.handleCreate)
	mux.HandleFunc("/clone", s.handleClone)
	mux.HandleFunc("/write", s.handleWrite)
	mux.HandleFunc("/truncate", s.handleTruncate)
	mux.HandleFunc("/delete", s.handleDelete)
	mux.HandleFunc("/batch", s.handleBatch)
	mux.HandleFunc("/read", s.handleRead)
	mux.HandleFunc("/stat", s.handleStat)
	mux.HandleFunc("/stats", s.handleStats)
	return mux
}

type mutationResponse struct {
	Revision int64 `json:"revision"`
}

type errorResponse struct {
	Error string `json:"error"`
}

// batchErrorResponse additionally reports which step failed; -1 means the
// batch itself was rejected (invalid shape or revision conflict) before any
// step ran.
type batchErrorResponse struct {
	Error string `json:"error"`
	Step  int    `json:"step"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, errorResponse{Error: err.Error()})
}

// statusFor maps domain errors to HTTP status codes.
func statusFor(err error) int {
	switch {
	case errors.Is(err, ErrInvalidName):
		return http.StatusBadRequest
	case errors.Is(err, ErrExists):
		return http.StatusConflict // distinct from a revision conflict
	case errors.Is(err, ErrConflict):
		return http.StatusConflict
	case errors.Is(err, ErrNotFound):
		return http.StatusNotFound
	case errors.Is(err, ErrOutOfRange):
		return http.StatusRequestedRangeNotSatisfiable
	case errors.Is(err, ErrQuota):
		return http.StatusInsufficientStorage
	case errors.Is(err, ErrInvalidBatch):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

// expectedRevision reads rev=... or X-Expected-Revision. A missing value is a
// malformed request; callers only use it on mutating routes.
func expectedRevision(r *http.Request) (int64, error) {
	raw := r.URL.Query().Get("rev")
	if h := r.Header.Get("X-Expected-Revision"); h != "" {
		raw = h
	}
	if raw == "" {
		return 0, errors.New("missing expected revision (rev or X-Expected-Revision)")
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n < 0 {
		return 0, errors.New("invalid expected revision")
	}
	return n, nil
}

func queryInt64(r *http.Request, key string, required bool, def int64) (int64, error) {
	raw := r.URL.Query().Get(key)
	if raw == "" {
		if required {
			return 0, errors.New("missing query parameter: " + key)
		}
		return def, nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0, errors.New("invalid query parameter: " + key)
	}
	return n, nil
}

func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	rev, err := expectedRevision(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	newRev, err := s.Vol.Create(r.URL.Query().Get("name"), rev)
	if err != nil {
		writeError(w, statusFor(err), err)
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

func (s *Server) handleClone(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	rev, err := expectedRevision(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	q := r.URL.Query()
	newRev, err := s.Vol.Clone(q.Get("src"), q.Get("dst"), rev)
	if err != nil {
		writeError(w, statusFor(err), err)
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

func (s *Server) handleWrite(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	rev, err := expectedRevision(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	offset, err := queryInt64(r, "offset", true, 0)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxWriteBytes)
	data, err := io.ReadAll(r.Body)
	if err != nil {
		if strings.Contains(err.Error(), "request body too large") {
			writeError(w, http.StatusRequestEntityTooLarge, err)
			return
		}
		writeError(w, http.StatusBadRequest, err)
		return
	}
	newRev, err := s.Vol.Write(r.URL.Query().Get("name"), offset, data, rev)
	if err != nil {
		writeError(w, statusFor(err), err)
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

func (s *Server) handleTruncate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	rev, err := expectedRevision(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	length, err := queryInt64(r, "length", true, 0)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	newRev, err := s.Vol.Truncate(r.URL.Query().Get("name"), length, rev)
	if err != nil {
		writeError(w, statusFor(err), err)
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

func (s *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	rev, err := expectedRevision(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	newRev, err := s.Vol.Delete(r.URL.Query().Get("name"), rev)
	if err != nil {
		writeError(w, statusFor(err), err)
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

// batchRequest is the body of POST /batch. The expected revision may also be
// supplied via rev= or X-Expected-Revision, which take precedence over the
// body field, matching the single-step endpoints.
type batchRequest struct {
	Revision *int64 `json:"revision"`
	Steps    []Step `json:"steps"`
}

// batchRevision resolves the single expected revision for a batch: query
// parameter or header first, then the request body.
func batchRevision(r *http.Request, req *batchRequest) (int64, error) {
	raw := r.URL.Query().Get("rev")
	if h := r.Header.Get("X-Expected-Revision"); h != "" {
		raw = h
	}
	if raw != "" {
		n, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || n < 0 {
			return 0, errors.New("invalid expected revision")
		}
		return n, nil
	}
	if req.Revision == nil || *req.Revision < 0 {
		return 0, errors.New("missing expected revision (rev, X-Expected-Revision or body revision)")
	}
	return *req.Revision, nil
}

func (s *Server) handleBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBatchBytes)
	var req batchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if strings.Contains(err.Error(), "request body too large") {
			writeError(w, http.StatusRequestEntityTooLarge, err)
			return
		}
		writeError(w, http.StatusBadRequest, errors.New("invalid batch body: "+err.Error()))
		return
	}
	rev, err := batchRevision(r, &req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	newRev, step, err := s.Vol.Batch(req.Steps, rev)
	if err != nil {
		writeJSON(w, statusFor(err), batchErrorResponse{Error: err.Error(), Step: step})
		return
	}
	writeJSON(w, http.StatusOK, mutationResponse{Revision: newRev})
}

func (s *Server) handleRead(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	offset, err := queryInt64(r, "offset", false, 0)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	length, err := queryInt64(r, "length", false, -1)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	data, rev, err := s.Vol.ReadWithRevision(r.URL.Query().Get("name"), offset, length)
	if err != nil {
		w.Header().Set("X-Revision", strconv.FormatInt(rev, 10))
		writeError(w, statusFor(err), err)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("X-Revision", strconv.FormatInt(rev, 10))
	w.Header().Set("X-Length", strconv.Itoa(len(data)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *Server) handleStat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	name := r.URL.Query().Get("name")
	if !ValidateName(name) {
		writeError(w, http.StatusBadRequest, ErrInvalidName)
		return
	}
	st := s.Vol.Stats()
	for i := range st.Files {
		if st.Files[i].Name == name {
			fi := st.Files[i]
			writeJSON(w, http.StatusOK, struct {
				FileInfo
				Revision int64 `json:"revision"`
			}{FileInfo: fi, Revision: st.Revision})
			return
		}
	}
	writeError(w, http.StatusNotFound, ErrNotFound)
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, errors.New("method not allowed"))
		return
	}
	writeJSON(w, http.StatusOK, s.Vol.Stats())
}
