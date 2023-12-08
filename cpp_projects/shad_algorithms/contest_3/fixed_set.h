// https://contest.yandex.ru/contest/29058/problems/

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <random>
#include <vector>

constexpr int64_t kPrime = 2'000'000'011;
constexpr size_t kDistributionUniformityDegree = 4;

namespace fixed_set_rng {
std::random_device random_device{};
std::mt19937_64 generator{random_device()};
}  // namespace fixed_set_rng

template <class T>
size_t PositiveMod(T value, T divisor) {
    if (divisor <= 0) {
        throw std::invalid_argument("Divisor should be positive");
    }
    auto mod = value % divisor;
    return mod < 0 ? mod + divisor : mod;
}

template <class T>
T Pow2(T value) {
    return value * value;
}

size_t SumUpElementsSquared(const std::vector<int32_t> &elements) {
    size_t sum = 0;
    for (auto element : elements) {
        sum += Pow2(element);
    }
    return sum;
}

class LinearHash {
public:
    LinearHash() = default;

    LinearHash(int64_t prime, int64_t first, int64_t second)
        : prime_{prime}, first_{first}, second_{second} {
    }

    size_t operator()(int64_t value) const {
        return PositiveMod(value * first_ + second_, prime_);
    }

private:
    int64_t prime_;
    int64_t first_;
    int64_t second_;
};

class UniversalLinearHash : public LinearHash {
public:
    UniversalLinearHash() = default;

    UniversalLinearHash(int64_t prime, int64_t first, int64_t second, size_t bucket_count)
        : LinearHash{prime, first, second}, bucket_count_{bucket_count} {
    }

    UniversalLinearHash(const LinearHash &linear_hash, size_t bucket_count)
        : LinearHash{linear_hash}, bucket_count_{bucket_count} {
    }

    size_t operator()(int64_t value) const {
        return PositiveMod(LinearHash::operator()(value), bucket_count_);
    }

    size_t GetBucketCount() const {
        return bucket_count_;
    }

private:
    size_t bucket_count_;
};

LinearHash GenerateRandomLinearHash(
    int64_t prime = kPrime,
    decltype(fixed_set_rng::generator) &generator = fixed_set_rng::generator) {

    std::uniform_int_distribution<int64_t> distribution{1, prime - 1};

    int64_t first = distribution(generator);
    int64_t second = distribution(generator);
    return {prime, first, second};
}

template <class Predicate>
UniversalLinearHash GenerateConformingUniversalHash(const std::vector<int32_t> &elements,
                                                    size_t bucket_count) {
    Predicate predicate;
    UniversalLinearHash hash_function;

    do {
        hash_function = UniversalLinearHash{GenerateRandomLinearHash(), bucket_count};
    } while (not predicate(elements, hash_function));

    return hash_function;
}

std::vector<int32_t> CalculateDistribution(UniversalLinearHash function,
                                           const std::vector<int32_t> &elements) {
    std::vector<int32_t> distribution(function.GetBucketCount());

    for (auto element : elements) {
        auto position = function(element);
        ++distribution[position];
    }
    return distribution;
}

class IsHashUniformOnElements {
public:
    bool operator()(const std::vector<int32_t> &elements, UniversalLinearHash function,
                    size_t distribution_uniformity = kDistributionUniformityDegree) const {

        auto distribution = CalculateDistribution(function, elements);
        return SumUpElementsSquared(distribution) / distribution_uniformity <= elements.size();
    }
};

class IsHashPerfectOnElements {
public:
    bool operator()(const std::vector<int32_t> &elements, UniversalLinearHash function) const {

        auto distribution = CalculateDistribution(function, elements);
        return SumUpElementsSquared(distribution) == elements.size();
    }
};

class SecondLevelHashTable {
public:
    void Initialize(const std::vector<int32_t> &elements) {
        hash_table_.clear();
        hash_table_.resize(Pow2(elements.size()));

        hash_function_ =
            GenerateConformingUniversalHash<IsHashPerfectOnElements>(elements, hash_table_.size());

        for (auto element : elements) {
            auto position = hash_function_(element);
            hash_table_[position].emplace(element);
        }
    }

    bool Contains(int32_t number) const {
        if (hash_table_.empty()) {
            return false;
        }
        auto position = hash_function_(number);
        return hash_table_[position] == number;
    }

private:
    std::vector<std::optional<int32_t>> hash_table_;
    UniversalLinearHash hash_function_;
};

class FixedSet {
public:
    void Initialize(const std::vector<int32_t> &elements) {
        hash_table_.clear();
        hash_table_.resize(elements.size());

        hash_function_ =
            GenerateConformingUniversalHash<IsHashUniformOnElements>(elements, hash_table_.size());

        auto buckets = DistributeElementsByBuckets(elements);

        for (size_t i = 0; i < buckets.size(); ++i) {
            hash_table_[i].Initialize(buckets[i]);
        }
    }

    bool Contains(int32_t number) const {
        if (hash_table_.empty()) {
            return false;
        }
        auto bucket_number = hash_function_(number);
        return hash_table_[bucket_number].Contains(number);
    }

private:
    std::vector<std::vector<int32_t>> DistributeElementsByBuckets(
        const std::vector<int32_t> &elements) const {

        std::vector<std::vector<int32_t>> buckets(hash_function_.GetBucketCount());

        for (auto element : elements) {
            auto bucket_index = hash_function_(element);
            buckets[bucket_index].push_back(element);
        }
        return buckets;
    }

    std::vector<SecondLevelHashTable> hash_table_;
    UniversalLinearHash hash_function_;
};
