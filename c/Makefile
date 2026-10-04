CC      ?= cc
CFLAGS  ?= -O3 -march=native
LDLIBS  = -lm

all: santa santa2

santa: santa.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

santa2: santa2.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

clean:
	rm -f santa santa2

.PHONY: all clean
