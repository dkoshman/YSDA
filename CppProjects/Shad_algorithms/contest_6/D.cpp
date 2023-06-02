#include <algorithm>
#include <array>
#include <cassert>
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
#include <variant>
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

struct RequestAddNode {
    int32_t parent = 0;
};

struct RequestRemoveNode {
    int32_t node = 0;
};

struct RequestFindLCA {
    int32_t first = 0;
    int32_t second = 0;
};

struct ResponseFindLCA {
    int32_t lca_node = 0;
};

using Request = std::variant<RequestAddNode, RequestRemoveNode, RequestFindLCA>;

class Input {
public:
    std::vector<Request> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t count_requests = 0;
        in >> count_requests;
        requests.reserve(count_requests);

        for (int32_t request_id = 0; request_id < count_requests; ++request_id) {
            char c = '\0';
            int32_t first = 0;
            in >> c >> first;
            --first;
            if (c == '+') {
                requests.emplace_back(RequestAddNode{first});

            } else if (c == '-') {
                requests.emplace_back(RequestRemoveNode{first});

            } else {
                int32_t second = 0;
                in >> second;
                --second;
                requests.emplace_back(RequestFindLCA{first, second});
            }
        }
    }
};

class Output {
public:
    std::vector<int32_t> lca_nodes;

    Output() = default;

    explicit Output(const std::vector<ResponseFindLCA> &responses) {
        lca_nodes.reserve(responses.size());
        for (auto response : responses) {
            lca_nodes.emplace_back(response.lca_node);
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            lca_nodes.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : lca_nodes) {
            out << item + 1 << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return lca_nodes != other.lca_nodes;
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

class NodeWithAncestors {
public:
    int32_t id = 0;
    int32_t next_maybe_alive_ancestor_or_self = 0;
    int32_t depth = 0;

    NodeWithAncestors() = default;

    explicit NodeWithAncestors(int32_t id, const NodeWithAncestors &parent)
        : id{id},
          next_maybe_alive_ancestor_or_self{id},
          depth{parent.depth + 1},
          galloping_ancestors_{parent.id} {};

    [[nodiscard]] bool IsAlive() const {
        return next_maybe_alive_ancestor_or_self == id;
    }

    std::optional<int32_t> GetNthGallopingAncestor(size_t galloping_ancestor_index) {
        MaybeExpand(galloping_ancestor_index);
        return galloping_ancestors_[galloping_ancestor_index];
    }

    void SetNthGallopingAncestor(size_t galloping_ancestor_index, int32_t ancestor) {
        MaybeExpand(galloping_ancestor_index);
        galloping_ancestors_[galloping_ancestor_index] = ancestor;
    }

private:
    std::vector<std::optional<int32_t>> galloping_ancestors_;

    void MaybeExpand(size_t size) {
        if (size >= galloping_ancestors_.size()) {
            galloping_ancestors_.resize(size * 2 + 1);
        }
    }
};

class GraphWithAncestors {
public:
    std::nullopt_t Respond(io::RequestAddNode request) {
        nodes_.emplace_back(static_cast<int32_t>(nodes_.size()), nodes_[request.parent]);
        return std::nullopt;
    }

    std::nullopt_t Respond(io::RequestRemoveNode request) {
        nodes_[request.node].next_maybe_alive_ancestor_or_self =
            nodes_[GetParent(request.node)].next_maybe_alive_ancestor_or_self;
        return std::nullopt;
    }

    io::ResponseFindLCA Respond(io::RequestFindLCA request) {
        auto first = request.first;
        auto second = request.second;

        if (nodes_[first].depth > nodes_[second].depth) {
            std::swap(first, second);
        }

        second = FindAncestorAtDepth(second, nodes_[first].depth);

        auto maybe_alive_lca = FindLcaOfNodesWithSameDepth(first, second);

        auto alive_lca = CompressAliveLinks(maybe_alive_lca);

        return {alive_lca};
    }

private:
    std::vector<NodeWithAncestors> nodes_{1};

    int32_t FindNthGallopingAncestor(int32_t node, size_t nth) {
        auto nth_ancestor = nodes_[node].GetNthGallopingAncestor(nth);

        if (not nth_ancestor) {
            auto n_minus_1th_ancestor = FindNthGallopingAncestor(node, nth - 1);
            nth_ancestor = FindNthGallopingAncestor(n_minus_1th_ancestor, nth - 1);
            nodes_[node].SetNthGallopingAncestor(nth, nth_ancestor.value());
        }

        return nth_ancestor.value();
    }

    int32_t GetParent(int32_t node) {
        return FindNthGallopingAncestor(node, 0);
    }

    int32_t FindAncestorAtDepth(int32_t node, int32_t depth) {
        if (depth > nodes_[node].depth) {
            throw std::invalid_argument{"Node's ancestors are not lower than itself."};
        }

        while (depth < nodes_[node].depth) {
            auto highest_ancestor_not_too_low =
                LargestPowerOfTwoNotGreaterThan(nodes_[node].depth - depth);
            node = FindNthGallopingAncestor(node, highest_ancestor_not_too_low);
        }

        return node;
    }

    int32_t FindLcaOfNodesWithSameDepth(int32_t first, int32_t second) {

        if (nodes_[first].depth != nodes_[second].depth) {
            throw std::invalid_argument{"Depths are not the same."};
        }

        int32_t max_relative_height_to_check_for_lca = nodes_[first].depth;

        while (first != second and max_relative_height_to_check_for_lca > 0) {

            auto n = LargestPowerOfTwoNotGreaterThan(max_relative_height_to_check_for_lca);
            auto first_nth_ancestor = FindNthGallopingAncestor(first, n);
            auto second_nth_ancestor = FindNthGallopingAncestor(second, n);

            if (first_nth_ancestor != second_nth_ancestor) {
                first = first_nth_ancestor;
                second = second_nth_ancestor;
                max_relative_height_to_check_for_lca = nodes_[first].depth;
            } else {
                max_relative_height_to_check_for_lca >>= 1;
            }
        }

        return first == second ? first : GetParent(first);
    }

    int32_t CompressAliveLinks(int32_t node) {
        std::vector<int32_t> dead_ancestors;

        while (not nodes_[node].IsAlive()) {
            dead_ancestors.emplace_back(node);
            node = nodes_[node].next_maybe_alive_ancestor_or_self;
        }

        for (auto dead_ancestor : dead_ancestors) {
            nodes_[dead_ancestor].next_maybe_alive_ancestor_or_self = node;
        }

        return node;
    }

    template <typename Integral>
    static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
        Integral log_two = 0;
        while (value >>= 1) {
            ++log_two;
        }
        return log_two;
    }
};

io::Output Solve(const io::Input &input) {
    GraphWithAncestors graph;
    std::vector<io::ResponseFindLCA> responses;

    for (auto request : input.requests) {
        auto visit = [&graph](auto request) -> std::optional<io::ResponseFindLCA> {
            return graph.Respond(request);
        };
        auto response = std::visit(visit, request);
        if (response) {
            responses.emplace_back(response.value());
        }
    }

    return io::Output{responses};
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

struct NodeGraph {
    NodeGraph() = default;
    std::vector<int32_t> children;
};

std::vector<NodeGraph> ConstructGraph(const std::vector<io::Request> &requests) {
    std::vector<NodeGraph> graph;
    graph.reserve(requests.size());
    graph.emplace_back();

    for (auto request : requests) {
        if (std::holds_alternative<io::RequestAddNode>(request)) {
            auto parent = std::get<io::RequestAddNode>(request).parent;
            graph[parent].children.push_back(static_cast<int32_t>(graph.size()));
            graph.emplace_back();
        }
    }
    return graph;
}

struct EulerNode {
    EulerNode(int32_t node_index, int32_t depth) : node_index{node_index}, depth{depth} {
    }
    int32_t node_index;
    int32_t depth;
};

struct Euler {
    std::vector<EulerNode> nodes;
    std::vector<int32_t> first_occurrence;
};

struct ComparatorNodeEulerByDepth {
    bool operator()(const EulerNode &lhv, const EulerNode &rhv) const {
        return lhv.depth < rhv.depth;
    }
};

void EulerRecursiveUtil(const std::vector<NodeGraph> &graph, std::vector<EulerNode> &euler,
                        std::vector<int32_t> &first_occurrence, int32_t node, int32_t depth) {
    if (first_occurrence[node] == 0) {
        first_occurrence[node] = euler.size();
    }

    euler.emplace_back(node, depth);

    for (auto child : graph[node].children) {
        EulerRecursiveUtil(graph, euler, first_occurrence, child, depth + 1);
        euler.emplace_back(node, depth);
    }
}

Euler ConstructEulerTraverse(const std::vector<NodeGraph> &graph) {
    std::vector<EulerNode> euler;
    euler.reserve(graph.size() * 2);
    std::vector<int32_t> first_occurrence(graph.size());

    EulerRecursiveUtil(graph, euler, first_occurrence, 0, 0);

    return {euler, first_occurrence};
}

std::vector<std::vector<int32_t>> PreprocessRMQ(const Euler &euler) {
    size_t log_two = 1;
    while (euler.nodes.size() >> log_two) {
        ++log_two;
    }
    std::vector<std::vector<int32_t>> rmq_preprocessed(log_two);

    rmq_preprocessed.front().reserve(euler.nodes.size());
    for (auto node : euler.nodes) {
        rmq_preprocessed.front().push_back(node.node_index);
    }

    for (size_t power_two = 1; power_two < log_two; ++power_two) {
        auto &vector = rmq_preprocessed[power_two];
        auto vector_size = euler.nodes.size() - (1 << power_two) + 1;
        vector.reserve(vector_size);

        auto previous_row_index = power_two - 1;
        for (size_t index = 0; index < vector_size; ++index) {
            auto first = rmq_preprocessed[previous_row_index][index];
            auto second = rmq_preprocessed[previous_row_index][index + (1 << previous_row_index)];

            if (euler.nodes[euler.first_occurrence[first]].depth <
                euler.nodes[euler.first_occurrence[second]].depth) {
                vector.push_back(first);
            } else {
                vector.push_back(second);
            }
        }
    }
    return rmq_preprocessed;
}

int32_t FindLCA(int32_t first_node, int32_t second_node,
                const std::vector<std::vector<int32_t>> &rmq_preprocessed, const Euler &euler) {
    auto first = euler.first_occurrence[first_node];
    auto second = euler.first_occurrence[second_node];
    if (first > second) {
        std::swap(first, second);
    }

    int32_t log_two = 0;
    while ((second - first + 1) >> log_two > 1) {
        ++log_two;
    }

    auto candidate_first = rmq_preprocessed[log_two][first];
    auto candidate_second = rmq_preprocessed[log_two][second - (1 << log_two) + 1];
    if (euler.nodes[euler.first_occurrence[candidate_first]].depth <
        euler.nodes[euler.first_occurrence[candidate_second]].depth) {
        return candidate_first;
    } else {
        return candidate_second;
    }
}

std::vector<int32_t> FindParents(const std::vector<NodeGraph> &graph) {
    std::vector<int32_t> parents(graph.size());
    for (size_t parent = 0; parent < graph.size(); ++parent) {
        for (auto child : graph[parent].children) {
            parents[child] = parent;
        }
    }
    return parents;
}

io::Output BruteForceSolveTwo(const io::Input &input) {
    auto graph = ConstructGraph(input.requests);
    auto parents = FindParents(graph);
    std::vector<bool> alive_vector(graph.size(), true);
    std::vector<int32_t> answers;

    for (auto request : input.requests) {
        if (std::holds_alternative<io::RequestRemoveNode>(request)) {

            auto node = std::get<io::RequestRemoveNode>(request).node;
            alive_vector[node] = false;
        } else if (std::holds_alternative<io::RequestFindLCA>(request)) {

            auto first = std::get<io::RequestFindLCA>(request).first;
            auto second = std::get<io::RequestFindLCA>(request).second;
            std::vector<int32_t> first_parents;
            std::vector<int32_t> second_parents;

            while (first != 0) {
                if (alive_vector[first]) {
                    first_parents.push_back(first);
                }
                first = parents[first];
            }
            first_parents.push_back(0);

            while (second != 0) {
                if (alive_vector[second]) {
                    second_parents.push_back(second);
                }
                second = parents[second];
            }
            second_parents.push_back(0);

            size_t index = 1;
            int32_t lca = 0;
            while (index <= first_parents.size() and index <= second_parents.size()) {
                if (*(first_parents.end() - index) == *(second_parents.end() - index)) {
                    lca = *(first_parents.end() - index);
                }
                ++index;
            }

            answers.push_back(lca);
        }
    }

    io::Output output;
    output.lca_nodes = answers;
    return output;
}

io::Output BruteForceSolve(const io::Input &input) {
    // Non-online solution.
    auto graph = ConstructGraph(input.requests);
    auto euler = ConstructEulerTraverse(graph);
    auto parents = FindParents(graph);
    auto rmq_preprocessed = PreprocessRMQ(euler);
    std::vector<bool> alive(graph.size(), true);
    graph.clear();
    std::vector<int32_t> answers;

    for (auto request : input.requests) {
        if (std::holds_alternative<io::RequestRemoveNode>(request)) {

            auto node = std::get<io::RequestRemoveNode>(request).node;
            alive[node] = false;
        } else if (std::holds_alternative<io::RequestFindLCA>(request)) {

            auto first = std::get<io::RequestFindLCA>(request).first;
            auto second = std::get<io::RequestFindLCA>(request).second;
            auto lca = FindLCA(first, second, rmq_preprocessed, euler);

            std::vector<int32_t> stack;
            while (not alive[lca]) {
                stack.push_back(lca);
                lca = parents[lca];
            }
            for (auto child : stack) {
                parents[child] = lca;
            }
            answers.push_back(lca);
        }
    }

    io::Output output;
    output.lca_nodes = answers;
    assert(not(output != BruteForceSolveTwo(input)));
    return output;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_requests = 1 + test_case_id;

    std::uniform_int_distribution<> node_dist(1, 1'000'000'007);
    std::discrete_distribution<> request_dist{0, 1, 2, 3};
    auto &engine = *rng::GetEngine();

    std::vector<io::Request> requests;
    requests.reserve(n_requests);
    std::vector<int32_t> alive;
    alive.push_back(0);
    int32_t node_count = 1;

    while (static_cast<int32_t>(requests.size()) < n_requests) {
        auto request_id = request_dist(engine);
        if (request_id < 2) {
            auto node_id = node_dist(engine) % static_cast<int32_t>(alive.size());
            requests.emplace_back(io::RequestAddNode{node_id});
            alive.push_back(node_count++);
        } else if (request_id == 2) {
            if (alive.size() == 1) {
                continue;
            }
            auto node_id = node_dist(engine) % static_cast<int32_t>(alive.size() - 1);
            requests.emplace_back(io::RequestRemoveNode{node_id + 1});
            alive.erase(std::remove(alive.begin(), alive.end(), node_id), alive.end());
        } else {
            auto first = node_dist(engine) % static_cast<int32_t>(alive.size());
            auto second = node_dist(engine) % static_cast<int32_t>(alive.size());
            requests.emplace_back(io::RequestFindLCA{first, second});
        }
    }

    io::Input input;
    input.requests = std::move(requests);
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(200'000);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, const std::string &expected) {
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
        "11\n"
        "+ 1\n"
        "+ 1\n"
        "+ 2\n"
        "? 2 3\n"
        "? 1 3\n"
        "? 2 4\n"
        "+ 4\n"
        "+ 4\n"
        "- 4\n"
        "? 5 6\n"
        "? 5 5\n",
        "0 0 1 1 4");

    timed_check.Check(
        "1\n"
        "? 1 1\n",
        "0");

    timed_check.Check(
        "1\n"
        "+ 1\n",
        "");

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
