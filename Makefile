CXX = g++
MPICXX = mpicxx
CXXFLAGS = -O3 -std=c++17

all: seq main

seq: seq.cpp
	$(CXX) $(CXXFLAGS) -o seq seq.cpp

main: main.cpp
	$(MPICXX) $(CXXFLAGS) -o main main.cpp

clean:
	rm -f seq main
