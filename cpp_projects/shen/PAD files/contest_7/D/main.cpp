#include <iostream>

class Writer {
public:
    virtual void write(const char* data, size_t len) {}
};

class BufferedWriter : public Writer {
public:
    const size_t _cap;
    size_t _size;
    char* _buf;
    BufferedWriter(size_t cap) : _cap(cap), _size(0) {
        _buf = new char[_cap];
    }
    ~BufferedWriter() {
        Writer::write(_buf, _size);
        delete [] _buf;
    }
    virtual void write(const char* data, size_t len) override {
        for (size_t i = 0; i < len; ++i) {
            if (_size == _cap) {
                Writer::write(_buf, _cap);
                _size = 0;
            }
            *(_buf + _size++) = *(data + i);
        }
    }
};
