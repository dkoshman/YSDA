#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

constexpr int32_t kPrime = 1'000'003;

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return seed;
}

void PrintSeed(std::ostream &ostream = std::cerr) {
    std::cerr << "Seed = " << GetSeed() << std::endl;
}

std::mt19937 *GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

int32_t PositiveMod(int64_t value, int32_t divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    int64_t mod = value % divisor;
    if (mod < 0) {
        mod += divisor > 0 ? divisor : -divisor;
    }
    return static_cast<int32_t>(mod);
}

class LinearHash {
public:
    explicit LinearHash(int32_t n_coefficients, int32_t prime,
                        std::mt19937 *engine = rng::GetEngine())
        : prime_{prime},
          engine_{engine},
          coefficient_distribution_{1, prime - 1},
          free_coefficient_{coefficient_distribution_(*engine)} {
        Resize(n_coefficients);
    }

    void Resize(size_t size) {
        coefficients_.reserve(size);
        while (coefficients_.size() < size) {
            coefficients_.emplace_back(coefficient_distribution_(*engine_));
        }
        coefficients_.resize(size);
    }

    template <class Iterator>
    int32_t operator()(Iterator begin, Iterator end) const {
        auto hash = static_cast<int64_t>(free_coefficient_);
        for (int32_t i = 0; i < end - begin; ++i) {
            auto value = static_cast<int64_t>(PositiveMod(*(begin + i), prime_));
            hash = PositiveMod(hash + value * coefficients_[i], prime_);
        }
        return static_cast<int32_t>(hash);
    }

private:
    int32_t prime_ = 0;
    std::mt19937 *engine_ = nullptr;
    std::uniform_int_distribution<int32_t> coefficient_distribution_;
    int32_t free_coefficient_ = 0;
    std::vector<int32_t> coefficients_;
};

struct Triangle {
    std::array<int32_t, 3> sides;

    Triangle(int32_t first_side, int32_t second_side, int32_t third_side)
        : sides{first_side, second_side, third_side} {
    }

    void Sort() {
        std::sort(sides.begin(), sides.end());
    }

    [[nodiscard]] Triangle GetSorted() const {
        auto triangle = *this;
        triangle.Sort();
        return triangle;
    }
};

class AreTrianglesSimilarOrMirroredSimilarComparator {
public:
    bool operator()(const Triangle &left, const Triangle &right) {
        auto left_sorted_sides = left.GetSorted().sides;
        auto right_sorted_sides = right.GetSorted().sides;

        bool is_first_to_second_ratio_equal =
            AreRatiosEqual(left_sorted_sides[0], left_sorted_sides[1], right_sorted_sides[0],
                           right_sorted_sides[1]);

        bool is_first_to_third_ratio_equal =
            AreRatiosEqual(left_sorted_sides[0], left_sorted_sides[2], right_sorted_sides[0],
                           right_sorted_sides[2]);

        return is_first_to_second_ratio_equal and is_first_to_third_ratio_equal;
    }

    [[nodiscard]] static bool AreRatiosEqual(int32_t first_numerator, int32_t first_denominator,
                                             int32_t second_numerator, int32_t second_denominator) {
        return static_cast<int64_t>(first_numerator) * second_denominator ==
               static_cast<int64_t>(first_denominator) * second_numerator;
    }
};

class TriangleSimilarityOrMirroredSimilarityHash {
public:
    explicit TriangleSimilarityOrMirroredSimilarityHash(int32_t prime_capacity)
        : linear_hash_{4, prime_capacity} {
    }

    int32_t operator()(const Triangle &triangle) const {
        auto sorted_sides = triangle.GetSorted().sides;
        auto [first_int, first_reverse_fraction] =
            ComputeProperFraction(sorted_sides[2], sorted_sides[1]);
        auto [second_int, second_reverse_fraction] =
            ComputeProperFraction(sorted_sides[2], sorted_sides[0]);

        std::array<int32_t, 4> triangle_stats{first_int, first_reverse_fraction, second_int,
                                              second_reverse_fraction};
        auto hash_value = linear_hash_(triangle_stats.begin(), triangle_stats.end());
        return hash_value;
    }

private:
    LinearHash linear_hash_;

    [[nodiscard]] static std::pair<int32_t, int32_t> ComputeProperFraction(int32_t numerator,
                                                                           int32_t denominator) {
        auto gcd = std::gcd(numerator, denominator);
        return {numerator / gcd, denominator / gcd};
    }
};

template <class Value, class Hash, class ValueEqualityComparator = std::equal_to<>>
class HashSet {
public:
    explicit HashSet(int32_t prime_table_size)
        : hash_{prime_table_size}, table_(prime_table_size), equality_comparator_{} {
    }

    bool Insert(const Value &value) {
        auto value_hash = hash_(value);
        if (HasValue(value, value_hash)) {
            return false;
        }
        table_[value_hash].push_back(value);
        return true;
    }

    [[nodiscard]] bool HasValue(const Value &value) {
        return HasValue(value, hash_(value));
    }

    [[nodiscard]] bool HasValue(const Value &value, int32_t value_hash) {
        const auto &values_with_same_hash = table_[value_hash];
        auto is_same_value = [value, this](auto &value_with_same_hash) -> bool {
            return equality_comparator_(value, value_with_same_hash);
        };
        return std::any_of(values_with_same_hash.begin(), values_with_same_hash.end(),
                           is_same_value);
    }

    [[nodiscard]] std::vector<Value> GetMaxBucket(bool print_max_bucket_contents = false) const {
        auto vector_size_comparator = [](const auto &left, const auto &right) -> bool {
            return left.size() < right.size();
        };
        return *std::max_element(table_.begin(), table_.end(), vector_size_comparator);
    }

    void PrintMaxBucket() const {
        auto max_bucket = GetMaxBucket();
        std::cerr << "Bucket of size " << max_bucket.size() << ":\n";
        for (auto t : max_bucket) {
            for (auto i : t.sides) {
                std::cerr << i << '\n';
            }
            std::cerr << '\n';
        }
        std::cerr << "End of bucket.\n";
    }

private:
    Hash hash_;
    std::vector<std::vector<Value>> table_;
    ValueEqualityComparator equality_comparator_;
};

namespace io {

class InputType {
public:
    std::vector<Triangle> triangles;

    InputType() = default;

    explicit InputType(std::istream &in) {
        int32_t triangle_count = 0;
        in >> triangle_count;
        triangles.reserve(triangle_count);

        for (int32_t i = 0; i < triangle_count; ++i) {
            int32_t first_side = 0;
            int32_t second_side = 0;
            int32_t third_side = 0;
            in >> first_side >> second_side >> third_side;
            triangles.emplace_back(first_side, second_side, third_side);
        }
    }
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(int32_t n_similarity_or_mirrored_similarity_classes)
        : n_similarity_or_mirrored_similarity_classes{n_similarity_or_mirrored_similarity_classes} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << n_similarity_or_mirrored_similarity_classes << '\n';
        return out;
    }

    bool operator!=(const OutputType &other) const {
        return n_similarity_or_mirrored_similarity_classes !=
               other.n_similarity_or_mirrored_similarity_classes;
    }

    int32_t n_similarity_or_mirrored_similarity_classes = 0;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType;

OutputType Solve(const InputType &input) {
    auto triangle_set = HashSet<Triangle, TriangleSimilarityOrMirroredSimilarityHash,
                                AreTrianglesSimilarOrMirroredSimilarComparator>{kPrime};
    int32_t n_triangle_similarity_classes = 0;
    for (const auto &triangle : input.triangles) {
        bool new_value = triangle_set.Insert(triangle);
        if (new_value) {
            ++n_triangle_similarity_classes;
        }
    }
    triangle_set.PrintMaxBucket();
    return OutputType{n_triangle_similarity_classes};
}

namespace test {

class WrongAnswerException : public std::exception {
public:
    WrongAnswerException() = default;

    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message{};
};

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

OutputType BruteForceSolve(const InputType &input) {
    throw NotImplementedError{};
}

Triangle GenRandomPseudoTriangle() {
    auto engine = rng::GetEngine();
    std::uniform_int_distribution<int32_t> distribution{1, 10};
    int32_t first = distribution(*engine);
    int32_t second = distribution(*engine);
    int32_t third = distribution(*engine);
    return {first, second, third};
}

std::pair<InputType, std::optional<OutputType>> GenerateRandomInputWithExpectedOutput(
    int32_t test_case_id) {

    int64_t n_classes = 1 << std::min(10, test_case_id);
    int64_t class_size = 1 << std::min(10, test_case_id);
    InputType input;
    input.triangles.reserve(class_size * n_classes);
    auto engine = rng::GetEngine();
    std::uniform_int_distribution<int32_t> distribution{1, 100'000'000};
    std::vector<Triangle> parents;
    AreTrianglesSimilarOrMirroredSimilarComparator similarity_comparator{};

    for (int iii = 0; iii < n_classes; ++iii) {
        auto triangle = GenRandomPseudoTriangle();
        triangle.Sort();
        bool is_new = true;
        for (const auto &parent : parents) {
            if (similarity_comparator(parent, triangle)) {
                is_new = false;
                break;
            }
        }
        if (not is_new) {
            continue;
        }
        parents.push_back(triangle);
        for (int j = 0; j < class_size; ++j) {
            auto coef = distribution(*engine);
            input.triangles.emplace_back(triangle.sides[0] * coef, triangle.sides[1] * coef,
                                         triangle.sides[2] * coef);
        }
    }

    return {input, OutputType{static_cast<int32_t>(parents.size())}};
}

std::pair<InputType, std::optional<OutputType>> StressTestIo() {
    int32_t n_triangles = 1'000'000;
    int32_t max_triangle_side = 100'000'000;

    InputType input;
    input.triangles.reserve(n_triangles);

    for (int i = 0; i < n_triangles; ++i) {
        auto largest_side = max_triangle_side - i;
        input.triangles.emplace_back(largest_side, largest_side - 1, largest_side - 2);
    }

    return {input, std::nullopt};
}

class TimeItInMilliseconds {
public:
    std::chrono::time_point<std::chrono::steady_clock> begin;
    std::chrono::time_point<std::chrono::steady_clock> end;

    TimeItInMilliseconds() {
        Begin();
    }

    void Begin() {
        begin = std::chrono::steady_clock::now();
    }

    int64_t End() {
        end = std::chrono::steady_clock::now();
        return Duration();
    }

    [[nodiscard]] int64_t Duration() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
};

int64_t Check(const InputType &input,
              const std::optional<OutputType> &expected_output_optional = std::nullopt) {

    TimeItInMilliseconds time;
    auto output = Solve(input);
    time.End();

    OutputType expected_output;
    if (expected_output_optional) {
        expected_output = expected_output_optional.value();
    } else {
        try {
            expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
            return time.Duration();
        }
    }

    if (output != expected_output) {
        std::stringstream ss;
        ss << "\n==============================Expected==============================\n"
           << expected_output
           << "\n==============================Received==============================\n"
           << output << "\n";
        throw WrongAnswerException{ss.str()};
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, int32_t expected) {
    std::stringstream input_stream{test_case};
    return Check(InputType{input_stream});
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

template <class Iterator>
Stats ComputeStats(Iterator begin, Iterator end) {
    auto size = end - begin;
    if (size == 0) {
        throw std::invalid_argument{"Empty container."};
    }

    auto mean = std::accumulate(begin, end, 0.0) / size;

    double std = 0;
    for (auto i = begin; i != end; ++i) {
        std += (*i - mean) * (*i - mean);
    }
    std = std::sqrt(std / size);

    auto max = static_cast<double>(*std::max_element(begin, end));

    return Stats{mean, std, max};
}

void Test() {
    rng::PrintSeed();

    Check(
        "3\n"
        "6 6 10\n"
        "15 25 15\n"
        "35 21 21\n",
        1);
    Check(
        "4\n"
        "3 4 5\n"
        "10 11 12\n"
        "6 7 8\n"
        "6 8 10\n",
        3);

    int32_t n_test_cases = 100;
    std::vector<int64_t> durations;
    durations.reserve(n_test_cases);
    TimeItInMilliseconds time_it;

    for (int32_t test_case_id = 0; test_case_id < n_test_cases; ++test_case_id) {
        auto [input, expected_output_optional] =
            GenerateRandomInputWithExpectedOutput(test_case_id);
        durations.emplace_back(Check(input, expected_output_optional));
    }

    auto [input, expected_output_optional] = StressTestIo();
    durations.emplace_back(Check(input, expected_output_optional));

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

    std::cout << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
