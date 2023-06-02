#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
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

namespace io {

struct WeightedEdge {
    int32_t from = 0;
    int32_t to = 0;
    int32_t weight = 0;

    WeightedEdge(int32_t from, int32_t to, int32_t weight) : from{from}, to{to}, weight{weight} {
    }
};

class Input {
public:
    std::vector<WeightedEdge> edges;
    int32_t n_nodes = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        size_t n_edges = 0;
        in >> n_nodes >> n_edges;
        edges.reserve(n_edges);
        for (size_t i = 0; i < n_edges; ++i) {
            size_t from = 0;
            size_t to = 0;
            int32_t weight = 0;
            in >> from >> to >> weight;
            edges.emplace_back(--from, --to, weight);
        }
    }
};

class Output {
public:
    int32_t smallest_max_edge_weight_in_spanning_tree = 0;

    Output() = default;

    explicit Output(int32_t smallest_max_edge_weight_in_spanning_tree)
        : smallest_max_edge_weight_in_spanning_tree{smallest_max_edge_weight_in_spanning_tree} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << smallest_max_edge_weight_in_spanning_tree;
        return out;
    }

    bool operator!=(const Output &other) const {
        return smallest_max_edge_weight_in_spanning_tree !=
               other.smallest_max_edge_weight_in_spanning_tree;
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

class DisjointedSetUnion {
public:
    explicit DisjointedSetUnion(int32_t n_nodes)
        : tree_size_(n_nodes, 1), path_compress_buffer_(n_nodes) {

        node_parent_.reserve(n_nodes);
        for (int32_t node = 0; node < n_nodes; ++node) {
            node_parent_.emplace_back(node);
        }
    }

    bool AreConnected(int32_t first, int32_t second) {
        return GetTreeRoot(first) == GetTreeRoot(second);
    }

    void Connect(int32_t first, int32_t second) {
        auto first_root = GetTreeRoot(first);
        auto second_root = GetTreeRoot(second);

        if (tree_size_[first_root] < tree_size_[second_root]) {
            std::swap(first_root, second_root);
        }

        node_parent_[second_root] = first_root;
        tree_size_[first_root] += tree_size_[second_root];
    }

private:
    std::vector<int32_t> node_parent_;
    std::vector<int32_t> tree_size_;
    std::vector<int32_t> path_compress_buffer_;

    [[nodiscard]] bool IsTreeRoot(int32_t node) const {
        return node_parent_[node] == node;
    }

    int32_t GetTreeRoot(int32_t node) {
        path_compress_buffer_.clear();

        while (not IsTreeRoot(node)) {
            path_compress_buffer_.emplace_back(node);
            node = node_parent_[node];
        }

        for (auto child : path_compress_buffer_) {
            node_parent_[child] = node;
        }

        return node;
    }
};

bool ComparatorEdgeWeight(const io::WeightedEdge &lhv, const io::WeightedEdge &rhv) {
    return lhv.weight < rhv.weight;
}

template <typename WeightedEdge, typename EdgeComparator>
std::vector<WeightedEdge> Kruscal(std::vector<WeightedEdge> edges, int32_t n_nodes,
                                  const EdgeComparator &comparator) {

    std::vector<WeightedEdge> minimal_spanning_tree_edges;
    minimal_spanning_tree_edges.reserve(n_nodes - 1);

    DisjointedSetUnion dsu{n_nodes};

    std::sort(edges.begin(), edges.end(), comparator);

    for (auto edge : edges) {
        if (not dsu.AreConnected(edge.from, edge.to)) {
            dsu.Connect(edge.from, edge.to);
            minimal_spanning_tree_edges.emplace_back(edge);
        }
    }

    return minimal_spanning_tree_edges;
}

io::Output Solve(const io::Input &input) {

    auto minimal_spanning_tree_edges = Kruscal(input.edges, input.n_nodes, ComparatorEdgeWeight);

    auto heaviest_edge = std::max_element(minimal_spanning_tree_edges.begin(),
                                          minimal_spanning_tree_edges.end(), ComparatorEdgeWeight);

    return io::Output{heaviest_edge->weight};
}

namespace test {

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

size_t GetRoot(std::vector<size_t> &forest, size_t node) {
    std::vector<size_t> stack;
    while (forest[node] != node) {
        stack.push_back(node);
        node = forest[node];
    }
    for (auto child : stack) {
        forest[child] = node;
    }
    return node;
}

bool IsConnected(std::vector<size_t> &forest, io::WeightedEdge edge) {
    return GetRoot(forest, edge.from) == GetRoot(forest, edge.to);
}

void Unite(std::vector<size_t> &forest, io::WeightedEdge edge) {
    forest[GetRoot(forest, edge.from)] = GetRoot(forest, edge.to);
}

int32_t Kruscal(io::Input input) {

    std::sort(input.edges.begin(), input.edges.end(),
              [](const io::WeightedEdge &lhv, const io::WeightedEdge &rhv) {
                  return lhv.weight < rhv.weight;
              });

    std::vector<size_t> directed_forest;
    directed_forest.reserve(input.n_nodes);
    for (int32_t i = 0; i < input.n_nodes; ++i) {
        directed_forest.push_back(i);
    }

    int32_t max_weight = 0;
    for (auto edge : input.edges) {
        if (not IsConnected(directed_forest, edge)) {
            max_weight = std::max(max_weight, edge.weight);
            Unite(directed_forest, edge);
        }
    }
    return max_weight;
}

io::Output BruteForceSolve(const io::Input &input) {
    return io::Output{Kruscal(input)};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
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
                ss << "\n================================Expected================================\n"
                   << expected_output
                   << "\n================================Received================================\n"
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

void Test() {
    rng::PrintSeed();

    TimedChecker timed_check;

    timed_check.Check(
        "4 4\n"
        "1 2 1\n"
        "2 3 2\n"
        "3 4 3\n"
        "4 1 4\n",
        3);

    timed_check.Check(
        "3 3\n"
        "1 2 3\n"
        "2 3 1\n"
        "3 1 2\n",
        2);

    timed_check.Check(
        "2 1\n"
        "1 2 2\n",
        2);

    timed_check.Check(
        "2 3\n"
        "1 2 2\n"
        "1 2 1\n"
        "1 2 3\n",
        1);

    timed_check.Check(
        "3 5\n"
        "1 2 2\n"
        "2 1 1\n"
        "1 2 3\n"
        "3 2 1\n"
        "2 3 4\n",
        1);

    std::cerr << "Basic tests OK:\n" << timed_check;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_check;
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
