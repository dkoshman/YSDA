////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief      Ring Buffer Smoke tests.
/// \author     Sergey Shershakov
/// \version    0.1.0
/// \date       26.06.2021
///             This code is for educational purposes of the course "Introduction
///             to programming" provided by the Faculty of Computer Science
///             at the Higher School of Economics.
////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cassert>

#include "ringbuf.hpp"

typedef RingBuffer<int> IntRingBuffer;
using namespace std;
template <typename T>
void print(T v) {
    cout << v << endl;
}
int main()
{
    // smoke tests for an empty buffer
    IntRingBuffer buf(5);

    // smoke tests for a buffer with elements
    assert(buf.getSize() == 5);
    assert(buf.getCount() == 0);
    assert(buf.isEmpty());
    assert(!buf.isFull());
    buf.push(5);
    print(buf.back());
    RingBuffer<double> B(2);
    print(B.getSize());
    B.push(1);
    B.push(2);
//    B.push(3);
    print(B.front());
    print(B.back());
    B.pop();
    print(B.getCount());
    B.pop();
    print(B.getCount());
    B.push(6);
    print(B.getCount());
    print(B.front());
    print(B.back());
    print(B.isEmpty());
    B.pop();
    print(B.isEmpty());
    B.push(7);
    B.push(5);
    print(B.back());
    RingBuffer<double> A(B);
    print(A.front());
    print(A.isEmpty());
    print("------------here-------------");
    B = A;
    print(B.front());
    print(A.front());
    print(B.back());
    print(A.back());
    RingBuffer<double> C = B;
    print(B.back());
    print(C.back());
    return 0;
}
