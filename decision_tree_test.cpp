#include "decision_tree.hpp"
#include <iostream>
using namespace std;

int main() {
    asher::Data data{
        {1, 2, 3, 4},
        //{4, 3, 2, 1},
        //{3, 2, 1, 4},
        //{2, 3, 4, 1}
    };
    
    asher::DataView view(data);
    
    cout << view << endl << endl;
    
    auto[left,right] = view.partition(1,1);
    
    cout << left << endl << endl;
    cout << right << endl << endl;
}
