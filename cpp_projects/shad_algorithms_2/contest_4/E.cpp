#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <complex>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace io {

class Input {
public:
    std::string first_string;
    std::string second_string;
    static constexpr char kAlphabet[] = "AGTC";

    Input() = default;

    explicit Input(std::istream &in) {
        in >> first_string >> second_string;
    }
};

class Output {
public:
    int32_t position_in_first_string_with_least_hamming_distance_to_second_string = 0;

    Output() = default;

    explicit Output(int32_t answer)
        : position_in_first_string_with_least_hamming_distance_to_second_string{answer} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << position_in_first_string_with_least_hamming_distance_to_second_string;
        return out;
    }

    bool operator!=(const Output &other) const {
        return position_in_first_string_with_least_hamming_distance_to_second_string !=
               other.position_in_first_string_with_least_hamming_distance_to_second_string;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

void SetUpFastIo() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

}  // namespace io

using Complex = std::complex<double>;

namespace utils {

template <class Dividend, class Divisor>
Divisor NonNegativeMod(Dividend value, Divisor divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    if (divisor < 0) {
        throw std::invalid_argument("Negative divisor.");
    }
    auto mod = value % divisor;
    if (mod < 0) {
        mod += divisor;
    }
    return static_cast<Divisor>(mod);
}

template <typename Integral>
static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
    if (value <= 0) {
        throw std::invalid_argument{"Non positive logarithm argument."};
    }
    Integral log_two = 0;
    while (value >>= 1) {
        ++log_two;
    }
    return log_two;
}

template <typename Number>
Number PowerOfTwo(Number value) {
    return value * value;
}

template <typename T, typename I>
std::vector<T> Take(const std::vector<T> &values, const std::vector<I> &indices) {
    std::vector<T> slice;
    slice.reserve(values.size());
    for (auto i : indices) {
        slice.emplace_back(values[i]);
    }
    return slice;
}

template <typename T>
std::vector<T> PointWiseMultiply(const std::vector<T> &first, const std::vector<T> &second) {
    if (first.size() != second.size()) {
        throw std::invalid_argument("Vector must have the same size.");
    }
    std::vector<T> product;
    product.reserve(first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        product.emplace_back(first[i] * second[i]);
    }
    return product;
}

template <typename T, typename V>
void PointWiseMultiply(std::vector<T> *first, V value) {
    for (auto &i : *first) {
        i *= value;
    }
}

template <typename T>
void ComputeVectorOfPowers(T value, int32_t up_to_power, std::vector<T> *out_ptr) {
    auto &out = *out_ptr;
    out.resize(up_to_power);
    out[0] = 1;
    for (size_t i = 1; i < out.size(); ++i) {
        out[i] = out[i - 1] * value;
    }
}

template <typename T>
std::vector<T> ComputeVectorOfPowers(T value, int32_t up_to_power) {
    std::vector<T> powers;
    ComputeVectorOfPowers(value, up_to_power, &powers);
    return powers;
}

template <typename T>
std::vector<T> ComplexVectorToType(const std::vector<Complex> &complex);

template <>
std::vector<int32_t> ComplexVectorToType(const std::vector<Complex> &complex) {
    std::vector<int32_t> abs_values;
    abs_values.reserve(complex.size());
    for (auto c : complex) {
        abs_values.emplace_back(std::round(c.real()));
    }
    return abs_values;
}

template <typename FIRST, typename SECOND>
bool IsClose(FIRST first, SECOND second, double precision = 1e-7) {
    return std::abs(first - second) < precision;
}

template <typename FIRST, typename SECOND>
bool AllClose(const std::vector<FIRST> &first, const std::vector<SECOND> &second,
              double precision = 1e-7) {
    if (first.size() != second.size()) {
        return false;
    }
    for (size_t i = 0; i < first.size(); ++i) {
        if (not IsClose(first[i], second[i], precision)) {
            return false;
        }
    }
    return true;
}

}  // namespace utils

namespace fft {

template <typename T>
T GetRootOfOneOfDegree(int32_t degree) {
    return degree % 2 == 0 ? -1 : 1;
}

template <>
Complex GetRootOfOneOfDegree(int32_t degree) {
    auto two_pi_div_n = static_cast<Complex::value_type>(2 * M_PI / degree);
    auto cos = std::cos(two_pi_div_n);
    auto sin = std::sin(two_pi_div_n);
    return {cos, sin};
}

std::vector<int32_t> BuildPermutation(int32_t power_of_two) {
    auto size = 1 << power_of_two;
    std::vector<int32_t> permutation(size);

    for (int32_t step = size, size_div_step = 1; step >= 2; step >>= 1, size_div_step <<= 1) {
        auto half_step = step >> 1;
        for (int32_t i = 0; i < size; i += step) {
            permutation[i + half_step] = permutation[i] + size_div_step;
        }
    }

    return permutation;
}

template <typename T>
inline void ButterflyTransform(T &first, T &second, T first_root_of_one, T second_root_of_one) {
    auto first_copy = first;
    first = first + first_root_of_one * second;
    second = first_copy + second_root_of_one * second;
}

template <typename T>
void ButterflyTransform(typename std::vector<T>::iterator begin, int32_t window_size,
                        int32_t window_offset, const std::vector<T> &roots_of_one) {
    auto end = begin + window_size;
    auto half_window_size = window_size / 2;
    auto size = roots_of_one.size();

    for (auto even = begin, odd = begin + half_window_size; odd != end; ++even, ++odd) {

        auto even_coefficient_index = window_offset + (even - begin);
        auto odd_coefficient_index = even_coefficient_index + half_window_size;
        auto n_times_root_was_squared = size / window_size;

        auto even_root_power_mod_size =
            utils::NonNegativeMod(even_coefficient_index * n_times_root_was_squared, size);
        auto odd_root_power_mod_size =
            utils::NonNegativeMod(odd_coefficient_index * n_times_root_was_squared, size);

        auto even_root = roots_of_one[even_root_power_mod_size];
        auto odd_root = roots_of_one[odd_root_power_mod_size];

        ButterflyTransform(*even, *odd, even_root, odd_root);
    }
}

template <typename T>
inline void FastButterflyTransform(T &first, T &second, T first_root_of_one) {
    auto first_root_times_second = first_root_of_one * second;
    second = first;
    second -= first_root_times_second;
    first += first_root_times_second;
}

template <typename T>
void FastButterflyTransform(typename std::vector<T>::iterator begin, int32_t window_size,
                            int32_t window_offset, const std::vector<T> &roots_of_one) {
    auto end = begin + window_size;
    auto half_window_size = window_size / 2;
    auto size = roots_of_one.size();
    auto n_times_root_was_squared = size / window_size;

    for (auto even = begin, odd = begin + half_window_size; odd != end; ++even, ++odd) {

        auto even_coefficient_index = window_offset + (even - begin);

        auto even_root_power_mod_size =
            utils::NonNegativeMod(even_coefficient_index * n_times_root_was_squared, size);

        auto even_root = roots_of_one[even_root_power_mod_size];

        FastButterflyTransform(*even, *odd, even_root);
    }
}

template <typename T>
void ButterflyTransform(std::vector<T> *vector, int32_t window_size,
                        const std::vector<T> &roots_of_one) {
    auto size = static_cast<int32_t>(roots_of_one.size());
    auto use_fast_butterfly_transform = utils::IsClose(roots_of_one[size / 2], -1.0);

    for (auto begin = vector->begin(); begin != vector->end(); begin += window_size) {
        auto offset = begin - vector->begin();
        if (use_fast_butterfly_transform) {
            FastButterflyTransform(begin, window_size, offset, roots_of_one);
        } else {
            ButterflyTransform(begin, window_size, offset, roots_of_one);
        }
    }
}

template <typename T>
void TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(std::vector<T> *coefficients,
                                                                  T root_of_one) {
    auto power_of_two =
        static_cast<int32_t>(utils::LargestPowerOfTwoNotGreaterThan(coefficients->size()));
    auto size = 1 << power_of_two;
    assert(size == static_cast<int32_t>(coefficients->size()));

    auto permutation = BuildPermutation(power_of_two);
    *coefficients = utils::Take(*coefficients, permutation);

    auto powers = utils::ComputeVectorOfPowers(root_of_one, size);

    for (int32_t window_size = 2; window_size <= size; window_size <<= 1) {
        ButterflyTransform(coefficients, window_size, powers);
    }
}

void FourierTransform(std::vector<Complex> *complex, Complex root,
                      bool scale_down_by_square_root_of_size = true) {

    auto size = static_cast<int32_t>(complex->size());
    if (size != 1 << static_cast<int32_t>((std::log2(size)))) {
        throw std::invalid_argument{"Size must be a power of two."};
    }

    TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(complex, root);

    if (scale_down_by_square_root_of_size) {
        utils::PointWiseMultiply(complex, 1.0 / std::sqrt(size));
    }
}

void FourierTransform(std::vector<Complex> *complex) {
    auto size = static_cast<int32_t>(complex->size());
    auto root = GetRootOfOneOfDegree<Complex>(size);
    FourierTransform(complex, root);
}

void InverseFourierTransform(std::vector<Complex> *complex) {
    auto size = static_cast<int32_t>(complex->size());
    auto inverse_root = 1.0 / GetRootOfOneOfDegree<Complex>(size);
    FourierTransform(complex, inverse_root);
}

}  // namespace fft

template <typename T>
std::vector<T> FastComputeConvolution(const std::vector<T> &first, const std::vector<T> &second) {
    std::vector<Complex> complex_first{first.begin(), first.end()};
    std::vector<Complex> complex_second{second.begin(), second.end()};
    auto original_convolution_size = first.size() + second.size() - 1;
    auto max_size = std::max(complex_first.size(), complex_second.size());

    if (max_size <= 1) {
        return utils::PointWiseMultiply(first, second);
    }

    auto power_of_two =
        static_cast<int32_t>(utils::LargestPowerOfTwoNotGreaterThan(max_size - 1)) + 2;
    auto convolution_size = 1 << power_of_two;
    complex_first.resize(convolution_size);
    complex_second.resize(convolution_size);

    fft::FourierTransform(&complex_first);
    fft::FourierTransform(&complex_second);

    auto convolution = utils::PointWiseMultiply(complex_first, complex_second);
    utils::PointWiseMultiply(&convolution, std::sqrt(convolution_size));

    fft::InverseFourierTransform(&convolution);

    convolution.resize(original_convolution_size);
    return utils::ComplexVectorToType<T>(convolution);
}

void ComputeLetterMask(const std::string &string, char letter, std::vector<int32_t> *out_ptr) {
    auto &out = *out_ptr;
    out.resize(string.size());
    for (size_t i = 0; i < string.size(); ++i) {
        out[i] = string[i] == letter;
    }
}

std::string ToLower(std::string string) {
    for (auto &c : string) {
        c = static_cast<char>(std::tolower(c));
    }
    return string;
}

io::Output Solve(const io::Input &input) {
    auto first_size = static_cast<int32_t>(input.first_string.size());
    auto second_size = static_cast<int32_t>(input.second_string.size());

    auto &first_string = input.first_string;
    auto second_string = input.second_string;
    std::reverse(second_string.begin(), second_string.end());

    std::vector<int32_t> similarities(first_size - second_size + 1);

    int32_t max_similarity = 0;
    int32_t max_similarity_position = 0;
    std::vector<int32_t> first_mask;
    std::vector<int32_t> second_mask;

    for (auto letter : io::Input::kAlphabet) {
        ComputeLetterMask(first_string, letter, &first_mask);
        ComputeLetterMask(second_string, letter, &second_mask);

        auto convolution = FastComputeConvolution(first_mask, second_mask);

        for (auto i = 0; i < static_cast<int32_t>(similarities.size()); ++i) {
            similarities[i] += convolution[i + second_size - 1];

            if (similarities[i] > max_similarity or
                (similarities[i] == max_similarity and i < max_similarity_position)) {
                max_similarity = similarities[i];
                max_similarity_position = i;
            }
        }
    }

    return io::Output{max_similarity_position + 1};
}

namespace test {

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 824484357;
    return seed;
}

void PrintSeed(std::ostream &ostream = std::cerr) {
    ostream << "Seed = " << GetSeed() << std::endl;
}

std::mt19937 *GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

namespace detail {

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

int64_t Timeit(const std::function<void()> &function) {
    detail::TimeItInMilliseconds time;
    function();
    time.End();
    return time.Duration();
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

std::ostream &operator<<(std::ostream &os, const Stats &stats) {
    os << "\tMean:\t" + std::to_string(stats.mean) << '\n'
       << "\tStd:\t" + std::to_string(stats.std) << '\n'
       << "\tMax:\t" + std::to_string(stats.max) << '\n';
    return os;
}

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

}  // namespace detail

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

struct TestIo {
    io::Input input;
    std::optional<io::Output> optional_expected_output;

    explicit TestIo(io::Input input) : input{std::move(input)} {
    }

    TestIo(io::Input input, io::Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

template <typename I>
std::optional<int32_t> ComputeHammingDistance(I first_begin, I first_end, I second_begin,
                                              I second_end) {
    if (first_end - first_begin != second_end - second_begin) {
        return std::nullopt;
    }
    int32_t distance = 0;
    for (; first_begin != first_end; ++first_begin, ++second_begin) {
        if (*first_begin != *second_begin) {
            ++distance;
        }
    }
    return distance;
}

io::Output BruteForceSolve(const io::Input &input) {
    if (input.first_string.size() > 1000) {
        throw NotImplementedError{};
    }
    auto second_size = static_cast<int32_t>(input.second_string.size());
    int32_t min_hamming_distance = INT32_MAX;
    int32_t min_position = 0;

    for (auto first_begin = input.first_string.begin();
         first_begin <= input.first_string.end() - second_size; ++first_begin) {
        auto distance =
            ComputeHammingDistance(first_begin, first_begin + second_size,
                                   input.second_string.begin(), input.second_string.end());
        if (distance.value() < min_hamming_distance) {
            min_position = static_cast<int32_t>(first_begin - input.first_string.begin());
            min_hamming_distance = distance.value();
        }
    }

    return io::Output{min_position + 1};
}

std::string GenerateRandomString(int32_t size) {
    std::uniform_int_distribution<int32_t> letters_dist{0, 3};
    std::string string;
    for (int32_t i = 0; i < size; ++i) {
        string += io::Input::kAlphabet[letters_dist(*rng::GetEngine())];
    }
    return string;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t string_size = std::min(200'000, 1 + test_case_id);

    std::uniform_int_distribution<int32_t> second_string_size_distribution{1, string_size};

    io::Input input;
    input.first_string = GenerateRandomString(string_size);
    input.second_string = GenerateRandomString(second_string_size_distribution(*rng::GetEngine()));

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(200'000);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, int32_t expected) {
        std::stringstream input_stream{test_case};
        io::Input input{input_stream};
        io::Output expected_output{expected};
        TestIo test_io{input, expected_output};
        Check(test_io);
    }

    io::Output TimedSolve(const io::Input &input) {
        io::Output output;
        auto solve = [&output, &input]() { output = Solve(input); };

        durations.emplace_back(detail::Timeit(solve));
        return output;
    }

    void Check(TestIo test_io) {
        auto output = TimedSolve(test_io.input);

        if (not test_io.optional_expected_output) {
            try {
                test_io.optional_expected_output = BruteForceSolve(test_io.input);
            } catch (const NotImplementedError &e) {
            }
        }

        if (test_io.optional_expected_output) {
            auto &expected_output = test_io.optional_expected_output.value();

            if (output != expected_output) {
                Solve(test_io.input);

                std::stringstream ss;
                ss << "\n================================Expected======================"
                      "==========\n"
                   << expected_output
                   << "\n================================Received======================"
                      "==========\n"
                   << output << "\n";

                throw WrongAnswerException{ss.str()};
            }
        }
    }
};

std::ostream &operator<<(std::ostream &os, TimedChecker &timed_checker) {
    if (not timed_checker.durations.empty()) {
        auto duration_stats =
            detail::ComputeStats(timed_checker.durations.begin(), timed_checker.durations.end());
        std::cerr << duration_stats;
        timed_checker.durations.clear();
    }
    return os;
}

std::vector<int32_t> GenerateRandomIntVector(int32_t size, int32_t max_abs_value) {
    std::vector<int32_t> vector(size);
    std::uniform_int_distribution<int32_t> values_distribution{-max_abs_value, max_abs_value};
    for (auto &value : vector) {
        value = values_distribution(*rng::GetEngine());
    }
    return vector;
}

template <typename T>
T ComputePolynomialAt(std::vector<T> &coefficients, T point) {
    T value = 0;
    T powered_point = 1;
    for (auto c : coefficients) {
        value += c * powered_point;
        powered_point *= point;
    }
    return value;
}

template <typename T>
void BruteForceTransformPolynomialCoefficientsToItsValuesAtRootsOfOne(
    std::vector<T> *coefficients, const std::vector<T> &roots_of_one) {
    std::vector<T> values;
    values.reserve(coefficients->size());
    for (auto root : roots_of_one) {
        values.emplace_back(ComputePolynomialAt(*coefficients, root));
    }
    *coefficients = values;
}

void TestFft() {

    for (int32_t test_case = 1; test_case < 100; ++test_case) {
        auto root = fft::GetRootOfOneOfDegree<int32_t>(test_case);
        auto powers = utils::ComputeVectorOfPowers(root, test_case);
        assert(utils::IsClose(powers.back() * root, 1.0));

        auto double_root = fft::GetRootOfOneOfDegree<double>(test_case);
        auto double_powers = utils::ComputeVectorOfPowers(double_root, test_case);
        assert(utils::IsClose(double_powers.back() * double_root, 1.0));

        auto complex_root = fft::GetRootOfOneOfDegree<std::complex<double>>(test_case);
        auto complex_powers = utils::ComputeVectorOfPowers(complex_root, test_case);
        assert(utils::IsClose(complex_powers.back() * complex_root, 1.0));
    }

    std::vector<int32_t> expected{0};
    assert(fft::BuildPermutation(0) == expected);

    expected = {0, 1};
    assert(fft::BuildPermutation(1) == expected);

    expected = {0, 2, 1, 3};
    assert(fft::BuildPermutation(2) == expected);

    expected = {0, 4, 2, 6, 1, 5, 3, 7};
    assert(fft::BuildPermutation(3) == expected);

    expected = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    assert(fft::BuildPermutation(4) == expected);

    for (int32_t test_case = 0; test_case < 100; ++test_case) {
        auto size = 1 << utils::LargestPowerOfTwoNotGreaterThan(1 + test_case);
        auto root = fft::GetRootOfOneOfDegree<int32_t>(size);
        auto roots = utils::ComputeVectorOfPowers(root, size);
        auto coefficients = GenerateRandomIntVector(size, size);
        auto coefficients_copy = coefficients;
        auto coefficients_copy_too = coefficients;
        BruteForceTransformPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients_copy, roots);
        fft::TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients_copy_too,
                                                                          root);
        if (not utils::AllClose(coefficients_copy, coefficients_copy_too)) {
            fft::TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients, root);
            assert(false);
        }
    }

    for (int32_t test_case = 0; test_case < 100; ++test_case) {
        auto size = 1 << utils::LargestPowerOfTwoNotGreaterThan(1 + test_case);
        auto root = fft::GetRootOfOneOfDegree<std::complex<double>>(size);
        auto roots = utils::ComputeVectorOfPowers(root, size);
        auto coefficients_int = GenerateRandomIntVector(size, size);
        std::vector<std::complex<double>> coefficients{coefficients_int.begin(),
                                                       coefficients_int.end()};
        auto coefficients_copy = coefficients;
        auto coefficients_copy_too = coefficients;
        BruteForceTransformPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients_copy, roots);
        fft::TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients_copy_too,
                                                                          root);
        if (not utils::AllClose(coefficients_copy, coefficients_copy_too)) {
            fft::TransformPaddedPolynomialCoefficientsToItsValuesAtRootsOfOne(&coefficients, root);
            assert(false);
        }
    }

    for (int32_t test_case = 0; test_case < 100; ++test_case) {
        auto size = 1 << utils::LargestPowerOfTwoNotGreaterThan(1 + test_case);
        auto coefficients_int = GenerateRandomIntVector(size, size);
        std::vector<Complex> coefficients{coefficients_int.begin(), coefficients_int.end()};
        auto coefficients_copy = coefficients;

        fft::FourierTransform(&coefficients);
        fft::InverseFourierTransform(&coefficients);

        assert(utils::AllClose(coefficients_copy, coefficients));
    }

    std::cerr << "FFT OK\n";
}

void Test() {
    rng::PrintSeed();

    TestFft();

    TimedChecker timed_checker;

    timed_checker.Check(
        "AGTCAGTC\n"
        "GTC",
        2);

    timed_checker.Check(
        "AAGGTTCC\n"
        "TCAA",
        5);

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    std::cerr << "OK\n";
}

}  // namespace test

int main(int argc, char *argv[]) {

    io::SetUpFastIo();

    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(io::Input{std::cin});
    }

    return 0;
}
