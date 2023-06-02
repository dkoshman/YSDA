#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_4_2_1(vector<int> &v){
    for (int &i : v)
        cout << i << ',';
    cout << endl;
}

void make_subtree_regular(vector<int> &subtree, size_t node, size_t tree_size) {
//    Invariant: subtree is regular everywhere except maybe for node.
//    A subtree runs through v[1..tree_size], a regular tree is a tree in which parent is no less than its children.
    while (((node*2 + 1 <= tree_size) and (subtree[node*2 + 1] > subtree[node]))
           or ((node*2 <= tree_size) and (subtree[node*2] > subtree[node]))) {
        if ((node*2 + 1 <= tree_size) and (subtree[node*2 + 1] >= subtree[node*2])) {
            swap(subtree[node], subtree[node*2 + 1]);
            node = node*2 + 1;
        } else {
            swap(subtree[node], subtree[node*2]);
            node = node*2;
        }
    }
}

int shen_4_2_1(){
    cout << "Sort an array of size n using tree sort" << endl;
    size_t n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    vector<int> v;
//    In order to keep indexing simple in a tree represented by flat array,
//    I increase size of array by 1 and ignore 1st element.
    ++n;
    mt19937 mt(time(nullptr));
    for (size_t i = 0; i < n; ++i)
        v.push_back(mt() % n);
    print_4_2_1(v);
    size_t tree_size = n - 1, node = n - 1;
    while (node != 0) {
        make_subtree_regular(v, node, tree_size);
        --node;
    }
//    Invariant: v[1] is the greatest element in v[1..tree_size], v[tree_size] <= v[tree_size + 1] <= ... <= v[n - 1].
    while (tree_size != 1) {
        swap(v[1], v[tree_size]);
        --tree_size;
        make_subtree_regular(v, 1, tree_size);
    }
    v.erase(v.begin());
    print_4_2_1(v);
    return 0;
}



