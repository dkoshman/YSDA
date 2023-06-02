#include <iostream>
#include <random>
#include <vector>
#include <ctime>


int shen_1_2_35(){
    int n = 0, m = 0;
    std::cout << "Given an array of size n, find max in all m consecutive elements. O(n), O(n)."
              << std::endl;
    std::cout << "Enter n, m: " << std::endl;
#ifdef QT_DEBUG
    n = 10; m = 3;
#else
    std::cin >> n >> m;
#endif
    std::vector<int> v;
    std::mt19937 mt(std::time(nullptr));
    for (int i = 0; i < n; ++i){
        v.push_back(mt() % n);
        std::cout << *(v.end() - 1) << ' ';
    }
    std::cout << std::endl;
    std::vector<int> left;
//    Array v is partitioned in consecutive subarrays of size m. Arrays left and right
//    contain cumulative max in these partitions, going from left and right sides of array v.
//    So each m-subarray is split in two parts, left contains the max of right side and right -
//    of left side. |>>--right-->>.<<--left--<<|
    left.push_back(v[0]);
    for (int i = 1; i < n; ++i){
        if (i % m == 0)
            left.push_back(v[i]);
        else
            left.push_back(std::max(v[i], left[i - 1]));
    }
    std::vector<int> right;
    right.resize(n);
    right[n - 1] = v[n - 1];
    for (int i = n - 2; i >= 0; --i){
        if (i % m == m - 1)
            right[i] = v[i];
        else
            right[i] = std::max(v[i], right[i + 1]);
    }
    for (int i = 0; i < n - m + 1; ++i){
        std::cout << (std::max(left[i + m - 1], right[i])) << ' ';
    }
    std::cout << std::endl;
    return 0;
}
