package main

import (
	"elasticdl.org/elasticdl/pkg/ps"
	"flag"
	"fmt"
	"log"
	"time"
)

var (
	// TODO: parse more args
	port = flag.Int("port", 2222, "The server port")
)

func main() {
	flag.Parse()
	address := fmt.Sprintf("localhost:%d", *port)
	serverDone := make(chan bool)
	ps.CreateServer(address, serverDone)
	log.Println("PS service started.")
	for {
		select {
		case done := <-serverDone:
			_ = done
			break
		default:
			// TODO: check master pod status and break loop if needed
			time.Sleep(time.Second * 30)
		}
	}
}
