ALL=decision_tree_test
CPPFLAGS=-Wall -Wextra -pedantic --std=c++17 -O0 -g -pthread

all: $(ALL)

decision_tree_test: decision_tree.hpp decision_tree_test.cpp
	g++ $(CPPFLAGS) decision_tree_test.cpp -o decision_tree_test

clean:
	rm -f $(ALL)
