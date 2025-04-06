CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra

all: create_small_query

create_small_query: create_small_query.cpp
	$(CXX) $(CXXFLAGS) -o create_small_query create_small_query.cpp

clean:
	rm -f create_small_query create_small_query.exe *.o

.PHONY: all clean