#include <iostream>
#include <chrono>

// Named constants, named types for logically different types
enum Descriptor : int { k_compute_timeout = 1023 };
enum class Device : bool { kGpu, kCpu };

// Named structs instead of pairs
struct Response {
    int pending_bytes;
    std::string payload;
};

// Separate namespace for implementation, non-interface functions
namespace detail {
void Compute(std::chrono::milliseconds);
}

// Descriptive template names
template <class Config> const Config& Get();

Response Compute(Descriptor, Device);

int main() {
    int attempts = 1;

    // Use assert to check logical consistency
    assert(attempts > 0);
    // Use throw if errors are expected
    if (attempts <= 0) {
        throw std::invalid_argument("");
    }
    Compute(k_compute_timeout, Device::kCpu);
    return 0;
}
