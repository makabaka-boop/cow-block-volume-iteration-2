// Command vfsd serves the virtual file volume over HTTP.
package main

import (
	"flag"
	"log"
	"net/http"

	vfs "vfs"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	srv := vfs.NewServer(vfs.New())
	log.Printf("vfsd: block=%d maxBlocks=%d listening on %s", vfs.BlockSize, vfs.MaxBlocks, *addr)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}
