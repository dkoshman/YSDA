#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

typedef std::vector<int64_t>::iterator It;

template <class T>
void Swap(T &i, T &j) {
    T temp = i;
    i = j;
    j = temp;
}

template <class T>
void Print(T begin, T end) {
    for (T i = begin; i < end; ++i) {
        std::cout << *i << ' ';
    }
    std::cout << '\n';
}

template <class Iterator, class Compare>  // should be function
class QuickSort {
public:
    QuickSort(Iterator begin, Iterator end, Compare comp) : comp_(comp) {
        Sort(begin, end);  // antipattern: logic in constructor
    }

    Iterator PickPivot(Iterator begin, Iterator end) const {
        //        std::random_device rd;
        //        std::mt19937 gen(rd());
        //        std::uniform_int_distribution<> distrib(0, end - begin - 1);
        //        Iterator pivot = begin + distrib(gen);
        //        return pivot;
        return end - 1;
    }

    Iterator Partition(Iterator begin, Iterator end, Iterator pivot) const {
        auto pivot_value = *pivot;
        bool left_empty = true;

        // Invariant: *x <= *pivot, x < begin; *x>= *pivot, x >= end
        while (begin < end) {
            if (comp_(*begin, pivot_value)) {
                ++begin;
                left_empty = false;
            } else if (comp_(pivot_value, *(end - 1))) {
                --end;
            } else if (begin == end - 1) {
                return left_empty ? ++begin : begin;
            } else {
                std::iter_swap(begin, end - 1);
                ++begin;
                --end;
                left_empty = false;
            }
        }

        return begin;
    }

    void Sort(Iterator begin, Iterator end) const {
        while (end - begin > 1) {
            Iterator pivot = PickPivot(begin, end);
            //            auto pvalue = *pivot;
            Iterator partition = Partition(begin, end, pivot);
            //            assert(partition != begin);
            //            assert(partition != end);
            //            for (auto it = begin; it < end; ++it) {
            //                if (((it < partition) and *it > pvalue) or
            //                    (it >= partition and *it < pvalue)) {
            //                        std::cout << pvalue << ',' << (partition - begin) << ":\n";
            //                        Print(begin, end);
            //                    }
            //            }
            Sort(begin, partition);
            begin = partition;
        }
    }

private:
    Compare comp_;
};



// It Partition(It begin, It end) {
//     auto pivot_value = *(begin + (end - begin) / 2);
//     bool left_empty = true;
//
//     // Invariant: *x <= *pivot, x < begin; *x>= *pivot, x >= end
//     while (begin < end) {
//         if (*begin < pivot_value) {
//             ++begin;
//             left_empty = false;
//         } else if (pivot_value < *(end - 1)) {
//             --end;
//         } else if (begin == end - 1) {
//             return left_empty ? ++begin : begin;
//         } else {
//             Swap(*begin, *(end - 1));
//             ++begin;
//             --end;
//             left_empty = false;
//         }
//     }
//
//     return begin;
// }
//It Partition(It begin, It end, It pivot) {
//    int64_t pivot_value = *pivot;
//    std::iter_swap(pivot, end - 1);
//
//    It lower_half = begin;
//    for (It it = begin; it < end - 1; ++it) {
//        if (*it <= pivot_value) {
//            Swap(*it, *lower_half);
//            ++lower_half;
//        }
//    }
//    std::iter_swap(lower_half, end - 1);
//
//    return lower_half == begin ? ++lower_half : lower_half;
//}

template <class Iterator, class Comparator>
Iterator PickPivot(Iterator begin, Iterator end, Comparator comparator) {
    Iterator mid = begin + (end - begin) / 2;
    --end;
    if (comparator(*mid, *begin)) {
        std::swap(mid, begin);
    }
    if (comparator(*mid, *end)) {
        return mid;
    }
    return comparator(*begin, *end) ? end : begin;

        return begin + (end - begin) / 2;
}

template <class Iterator, class Comparator>
Iterator Partition(Iterator begin, Iterator end, Iterator pivot, Comparator comparator) {

    auto pivot_value = *pivot;
    Iterator former_begin = begin;
    --end;

    while (begin <= end) {
        if (comparator(*begin, pivot_value)) {
            ++begin;
        } else if (comparator(pivot_value, *end)) {
            --end;
        } else {
            std::iter_swap(begin, end);
            ++begin;
            --end;
        }
    }

    if (begin == end + 2 and begin - 1 != former_begin) {
        return begin - 1;
    }
    return begin;
}

template <class Iterator, class Comparator>
void Sort(Iterator begin, Iterator end, Comparator comparator) {
    while (end - begin > 1) {

        Iterator pivot = PickPivot(begin, end, comparator);
        Iterator partition = Partition(begin, end, pivot, comparator);

        if (partition - begin < end - partition) {
            Sort(begin, partition, comparator);
            begin = partition;
        } else {
            Sort(partition, end, comparator);
            end = partition;
        }
    }
}

class Timeit {
public:
    Timeit(std::string annotation) {
        std::cout << annotation;
        begin_ = std::chrono::steady_clock::now();
    }
    ~Timeit() {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin_).count()
                  << "[µs]\n";
    }
    std::chrono::steady_clock::time_point begin_;
};

template <class Integer, class Compare>
bool Check(std::vector<Integer> vector, Compare comp) {
    auto vector_copy = vector;

    {
        Timeit time("my sort\t");
        Sort(vector.begin(), vector.end(), comp);
    }
    {
        Timeit time("lb sort\t");
        std::sort(vector_copy.begin(), vector_copy.end(), comp);
    }

    return vector == vector_copy;
}

template <class Integer>
bool Check(std::vector<Integer> vector) {
    return Check(vector, std::less<Integer>());
}

void Test() {

    //    assert(Check<int>({5, 4, 7, 1, 2, 5, 9, 123}));
    //    assert(Check<int>({9}));
    //    assert(Check<int>({}));

    std::random_device rd;
    std::mt19937 gen(rd());
    int64_t limit = 1;
    limit <<= 20;
    for (int64_t size = 1; size < limit; size <<= 1) {
        std::uniform_int_distribution<> distrib(-size, size);
        std::vector<int64_t> vector(size);
        for (auto &value : vector) {
            value = distrib(gen);
        }
        assert(Check(vector));
        std::cout << size << " - OK\n";
    }

    limit <<= 10;
    for (int64_t size = 1; size < limit; size <<= 1) {
        std::vector<int64_t> vector(size, size);

        assert(Check(vector));
        std::cout << size << " - equal elements - OK\n";
    }


    for (int64_t size = 1; size < limit; size <<= 1) {
        std::vector<int64_t> vector;
        vector.reserve(size);
        for (int64_t i = size; i > 0; --i) {
            vector.push_back(i);
        }
        assert(Check(vector));
        std::cout << size << " - reverse sorted - OK\n";
    }

    std::cout << "OK\n";
}

It Partition(It begin, It end, It pivot, std::vector<int> &v) {
    auto pivot_value = *pivot;
    --end;
    while (begin < end) {
        if (*begin < pivot_value) {
            ++begin;
        } else if (pivot_value < *end) {
            --end;
        } else {
            std::iter_swap(begin, end);
            ++begin;
            --end;
        }
    }
    if (*begin >= pivot_value) {
        return begin;
    }
    return ++begin;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    Test();
    return 0;
}
