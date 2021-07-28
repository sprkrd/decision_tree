#include "decision_tree.hpp"
#include "running_stats.hpp"
#include <iostream>
using namespace std;

int main() {
    asher::RunningStats stats;

    stats.push(49.5381252510948);
    stats.push(14.0715803666405);
    stats.push(76.1559544607964);
    stats.push(13.1909291554923);
    stats.push(42.8804564618735);
    stats.push(49.8856002549371);
    stats.push(29.9059441375031);
    cout << stats.mean() << endl;
    cout << stats.variance() << endl;

    stats.pop(49.5381252510948);
    stats.pop(14.0715803666405);
    cout << stats.mean() << endl;
    cout << stats.variance() << endl;

    stats.push(49.5381252510948);
    cout << stats.mean() << endl;
    cout << stats.variance() << endl;


    //asher::Data data{
        //{1, 2, 3, 4},
        ////{4, 3, 2, 1},
        ////{3, 2, 1, 4},
        ////{2, 3, 4, 1}
    //};
    
    //asher::DataView view(data);
    
    //cout << view << endl << endl;
    
    //auto[left,right] = view.partition(1,1);
    
    //cout << left << endl << endl;
    //cout << right << endl << endl;
}
