# tools/zmq-wire-bench/Makefile — build the isolated ZMQ wire benchmark producer.
# Public Domain (The Unlicense).
producer: producer.cpp
	g++ -O2 -g -std=c++17 producer.cpp -o producer -lzmq -lpthread
.PHONY: clean
clean:
	rm -f producer perf-wire.data
