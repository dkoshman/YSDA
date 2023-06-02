#include <iostream>
#include <string>
#include <memory>


class Serializer {
public:
    virtual void beginArray() {};
    virtual void addArrayItem(const std::string &) {};
    virtual void endArray() {};
};

class JsonSerializer : public Serializer {
    bool first = true;
public:
    JsonSerializer() {
        first = true;
    }

    void beginArray() {
        if (not first)
            std::cout << ',';
        std::cout << '[';
        first = true;
    }
    void addArrayItem(const std::string &s) {
        if (not first)
            std::cout << ',';
        std::cout << '"' << s << '"';
        first = false;
    }
    void endArray() {
        std::cout << ']';
        first = false;
    }
};

int main()
{
    return 0;
}
