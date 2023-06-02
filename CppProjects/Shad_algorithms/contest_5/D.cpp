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
    return 2019663117;
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

template <int32_t max_weight>
struct DirectedWeightedEdge {

    int32_t from = 0;
    int32_t to = 0;
    int32_t weight = 0;

    DirectedWeightedEdge(int32_t from, int32_t to, int32_t weight)
        : from{from}, to{to}, weight{weight} {
        if (weight < 0 or max_weight < weight) {
            throw std::invalid_argument{"Invalid weight."};
        }
    }
};

struct ShortestPathLengthRequest {
    int32_t from = 0;
    int32_t to = 0;

    ShortestPathLengthRequest(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

constexpr int32_t kMaxEdgeWeight = 2;

class Input {
public:
    int32_t n_nodes = 0;
    std::vector<DirectedWeightedEdge<kMaxEdgeWeight>> edges;
    std::vector<ShortestPathLengthRequest> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> n_nodes;
        int32_t n_edges = 0;
        in >> n_edges;

        edges.reserve(n_edges);
        for (int32_t edge = 0; edge < n_edges; ++edge) {
            int32_t from;
            int32_t to;
            int32_t weight;

            in >> from >> to >> weight;

            --from;
            --to;

            edges.emplace_back(from, to, weight);
        }

        int32_t n_requests = 0;
        in >> n_requests;
        requests.reserve(n_requests);
        for (int32_t request_id = 0; request_id < n_requests; ++request_id) {
            int32_t from;
            int32_t to;

            in >> from >> to;

            --from;
            --to;

            requests.emplace_back(from, to);
        }
    }
};

class Output {
public:
    std::vector<std::optional<int32_t>> request_responses;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            if (item == -1) {
                request_responses.emplace_back(std::nullopt);
            } else {
                request_responses.emplace_back(item);
            }
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : request_responses) {
            out << (item ? item.value() : -1) << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return request_responses != other.request_responses;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

}  // namespace io

using io::Input, io::Output;

namespace interface {

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

class VirtualBaseClass {
public:
    ~VirtualBaseClass() = default;
};

class ResizeInterface : public VirtualBaseClass {
public:
    virtual void Resize(size_t size) = 0;

    [[nodiscard]] virtual size_t Size() const = 0;
};

template <class NodeId = int32_t>
class ConnectByNodeIdsInterface : public VirtualBaseClass {
public:
    virtual void Connect(NodeId from, NodeId to) = 0;
};

template <class Edge>
class ConnectByEdgesInterface : public VirtualBaseClass {
public:
    virtual void AddEdge(Edge edge) = 0;
};

template <class Node, class NodeId = int32_t>
class NodeByIdInterface : public VirtualBaseClass {
public:
    virtual Node &operator[](NodeId node_id) = 0;

    virtual Node operator[](NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class AdjacentByIdInterface : public VirtualBaseClass {
public:
    //    [[nodiscard]] virtual NodeIdIterable NodeIdsAdjacentTo(NodeId node_id) const = 0;
};

template <class NodeId = int32_t>
class VisitAdjacentByIdInterface : public VirtualBaseClass {
public:
    virtual void VisitNodeIdsAdjacentTo(NodeId node_id,
                                        const std::function<void(NodeId node_id)> &function) = 0;
};

template <class NodeState>
class StateInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetState() const = 0;

    virtual void SetState(NodeState state) = 0;
};

template <class NodeState, class NodeId = int32_t>
class StateByIdInterface : public VirtualBaseClass {
public:
    [[nodiscard]] virtual NodeState GetNodeState(NodeId node_id) const = 0;

    virtual void SetNodeState(NodeId node_id, NodeState state) = 0;

    virtual ~StateByIdInterface() = default;
};

class GraphTraversal {
public:
    template <class Node>
    void OnNodeEnter(Node &node) {
    }

    template <class Node>
    void OnEdgeDiscovery(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeDiscovery(Edge edge) {
    }

    template <class Node>
    void OnEdgeTraverse(Node &from, Node &to) {
    }

    template <class Edge>
    void OnEdgeTraverse(Edge &edge) {
    }

    template <class Node>
    void OnNodeExit(Node &node) {
    }

    template <class Node>
    void SetNodeStateEntered(Node &node) {
    }

    template <class Node>
    void SetNodeStateExited(Node &node) {
    }

    template <class Node>
    bool IsNodeUnvisited(Node &node) {
        throw NotImplementedError{};
    }

    template <class Node, class NodeIterable = std::vector<Node> &>
    NodeIterable NodesAdjacentTo(Node &node) {
        throw NotImplementedError{};
    }
};

template <class Value>
class GeneratorInterface : interface::VirtualBaseClass {
    virtual std::optional<Value> Next() = 0;
};

}  // namespace interface

namespace implementation {

struct DirectedEdge {
    int32_t from = 0;
    int32_t to = 0;

    DirectedEdge() = default;

    DirectedEdge(int32_t from, int32_t to) : from{from}, to{to} {
    }
};

struct UndirectedEdge : public DirectedEdge {
    using DirectedEdge::DirectedEdge;
};

class DirectedConnectByNodeIdsImplementation : public interface::ConnectByNodeIdsInterface<int32_t>,
                                               public interface::ResizeInterface {
public:
    std::vector<std::vector<int32_t>> adjacent_ids_table;

    void Resize(size_t size) override {
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_ids_table.size();
    }

    void Connect(int32_t from, int32_t to) override {
        adjacent_ids_table[from].push_back(to);
    }
};

class UndirectedConnectByNodeIdsImplementation : public DirectedConnectByNodeIdsImplementation {
    void Connect(int32_t from, int32_t to) override {
        DirectedConnectByNodeIdsImplementation::Connect(from, to);
        if (from != to) {
            DirectedConnectByNodeIdsImplementation::Connect(to, from);
        }
    }
};

template <class Edge, class = std::enable_if_t<std::is_base_of_v<DirectedEdge, Edge>>>
class ConnectByEdgesImplementation : public interface::ConnectByEdgesInterface<Edge> {
public:
    std::vector<std::vector<Edge>> edges_table;

    void Resize(size_t size) override {
        edges_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return edges_table.size();
    }

    void AddEdge(Edge edge) override {
        AddDirectedEdge(edge);

        if (std::is_base_of_v<UndirectedEdge, Edge> and edge.from != edge.to) {
            std::swap(edge.from, edge.to);
            AddDirectedEdge(edge);
        }
    }

    void AddDirectedEdge(Edge edge) {
        edges_table[edge.from].emplace_back(std::move(edge));
    }
};

template <class Node>
class NodeByIdImplementation : public interface::NodeByIdInterface<Node, int32_t> {
public:
    std::vector<Node> nodes;

    void Resize(size_t size) override {
        nodes.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return nodes.size();
    }

    Node &operator[](int32_t node_id) override {
        return nodes[node_id];
    }

    Node operator[](int32_t node_id) const override {
        return nodes[node_id];
    }
};

class AdjacentByIdImplementation : public interface::AdjacentByIdInterface<int32_t>,
                                   public interface::ResizeInterface {
public:
    std::vector<std::vector<int32_t>> adjacent_ids_table;

    void Resize(size_t size) override {
        adjacent_ids_table.resize(size);
    }

    [[nodiscard]] size_t Size() const override {
        return adjacent_ids_table.size();
    }

    [[nodiscard]] const std::vector<int32_t> &NodeIdsAdjacentTo(int32_t node_id) const {
        return adjacent_ids_table[node_id];
    }
};

class DepthFirstSearch {
public:
    template <class Node>
    void Traverse(Node &starting_node) {
        TraverseRecursive(starting_node);
    }

private:
    template <class Node>
    void TraverseRecursive(Node &node) {

        OnNodeEnter(node);
        SetNodeStateEntered(node);

        for (auto &adjacent_node : NodeIdsAdjacentTo(node)) {

            OnEdgeDiscovery(node, adjacent_node);

            if (IsNodeUnvisited(adjacent_node)) {

                OnEdgeTraverse(node, adjacent_node);
                TraverseRecursive(adjacent_node);
            }
        }

        OnNodeExit(node);
        SetNodeStateExited(node);
    }
};

template <class GraphTraversal, class Node,
          class = std::enable_if_t<std::is_base_of_v<interface::GraphTraversal, GraphTraversal>>>
void BreadthFirstSearch(GraphTraversal *graph_traversal, std::deque<Node> starting_nodes_queue) {

    auto &queue = starting_nodes_queue;

    while (not queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateEntered(node);

        for (auto &adjacent_node : graph_traversal->NodesAdjacentTo(node)) {

            graph_traversal->OnEdgeDiscovery(node, adjacent_node);

            if (graph_traversal->IsNodeUnvisited(adjacent_node)) {

                graph_traversal->OnEdgeTraverse(node, adjacent_node);
                queue.emplace_back(adjacent_node);
            }
        }

        graph_traversal->OnNodeEnter(node);
        graph_traversal->SetNodeStateExited(node);
    }
}

template <class Generator,
          class = std::enable_if_t  // @formatter:off
          <std::is_base_of_v<       // @formatter:on
              interface::GeneratorInterface<typename Generator::Value>, Generator>>>
class GeneratorIterable : interface::VirtualBaseClass {
public:
    using Value = typename Generator::Value;

    struct Iterator {
        using IteratorCategory = std::forward_iterator_tag;
        using DifferenceType = std::ptrdiff_t;
        using ValueType = Value;
        using Pointer = Value *;
        using Reference = Value &;

        Iterator() = default;

        explicit Iterator(Generator generator) : optional_generator_{std::move(generator)} {
            PrecomputeValue();
        }

        ValueType operator*() const {
            return optional_value_.value();
        }

        Iterator &operator++() {
            if (IsGeneratorExhausted()) {
                throw std::runtime_error{"Generator is exhausted."};
            }
            PrecomputeValue();
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsGeneratorExhausted() and right.IsGeneratorExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    private:
        std::optional<Value> optional_value_;
        std::optional<Generator> optional_generator_;

        [[nodiscard]] bool IsGeneratorExhausted() const {
            return not optional_value_;
        }

        void PrecomputeValue() {
            optional_value_ = optional_generator_->Next();
        }
    };

    explicit GeneratorIterable(Generator generator) : generator_{std::move(generator)} {
    }

    Iterator begin() {
        return Iterator{generator_};
    }

    Iterator end() {
        return Iterator{};
    }

private:
    Generator generator_;
};

}  // namespace implementation

template <int32_t max_weight>
class DirectedGraphWithLimitedWeights
    : public interface::ConnectByEdgesInterface<io::DirectedWeightedEdge<max_weight>> {

public:
    explicit DirectedGraphWithLimitedWeights(size_t size)
        : edges_table_by_id_and_weight(size), incremental_traversal_up_to_current_distance_{this} {
    }

    void AddEdge(io::DirectedWeightedEdge<max_weight> edge) override {
        edges_table_by_id_and_weight[edge.from][edge.weight].emplace_back(edge.to);
    }

    std::optional<int32_t> ComputeShortestDistanceFromAtoB(int32_t from, int32_t to) {

        ResetDistances();

        SetSource(from);

        while (not HasFoundShortestDistanceFromSourceTo(to)) {

            IncrementallyTraverseGraphFromSourceUpToCurrentDistance();

            ++current_distance_from_source_;

            FinishProcessingNodesNotFurtherFromSourceThanCurrentDistanceMinusMaxWeight();
        }

        return distance_from_source_by_id_[to];
    }

    std::vector<std::array<std::vector<int32_t>, max_weight + 1>> edges_table_by_id_and_weight;

private:
    void ResetDistances() {
        distance_from_source_by_id_.clear();
        distance_from_source_by_id_.resize(edges_table_by_id_and_weight.size());
        nodes_being_processed_.clear();
        current_distance_from_source_ = 0;
    }

    void SetSource(int32_t source) {
        nodes_being_processed_.emplace_back(source);
        distance_from_source_by_id_[source] = current_distance_from_source_;
    }

    bool HasFoundShortestDistanceFromSourceTo(int32_t node) {
        return distance_from_source_by_id_[node] or nodes_being_processed_.empty();
    }

    void IncrementallyTraverseGraphFromSourceUpToCurrentDistance() {
        implementation::BreadthFirstSearch(&incremental_traversal_up_to_current_distance_,
                                           nodes_being_processed_);
    }

    void FinishProcessingNodesNotFurtherFromSourceThanCurrentDistanceMinusMaxWeight() {
        while (not nodes_being_processed_.empty() and
               current_distance_from_source_ -
                       distance_from_source_by_id_[nodes_being_processed_.front()].value() >
                   max_weight) {
            nodes_being_processed_.pop_front();
        }
    }

    class IncrementalTraversalUpToCurrentDistance : public interface::GraphTraversal {
    public:
        using Node = int32_t;

        explicit IncrementalTraversalUpToCurrentDistance(DirectedGraphWithLimitedWeights *graph)
            : graph{*graph} {
        }

        bool IsNodeUnvisited(Node &node) {
            return not graph.distance_from_source_by_id_[node];
        }

        std::vector<Node> &NodesAdjacentTo(Node &node) {
            auto edge_weight_to_traverse = graph.current_distance_from_source_ -
                                           graph.distance_from_source_by_id_[node].value();
            return graph.edges_table_by_id_and_weight[node][edge_weight_to_traverse];
        }

        void OnEdgeTraverse(Node &from, Node &to) {
            graph.distance_from_source_by_id_[to] = graph.current_distance_from_source_;
            graph.nodes_being_processed_.emplace_back(to);
        }

        DirectedGraphWithLimitedWeights &graph;
    };

    std::vector<std::optional<int32_t>> distance_from_source_by_id_;
    int32_t current_distance_from_source_ = 0;
    std::deque<int32_t> visited_nodes_further_from_source_than_current_distance_minus_max_weight_;
    std::deque<int32_t> &nodes_being_processed_ =
        visited_nodes_further_from_source_than_current_distance_minus_max_weight_;
    IncrementalTraversalUpToCurrentDistance incremental_traversal_up_to_current_distance_;
};

Output Solve(const Input &input) {
    DirectedGraphWithLimitedWeights<io::kMaxEdgeWeight> graph(input.n_nodes);

    for (auto edge : input.edges) {
        graph.AddEdge(edge);
    }

    Output output;
    for (auto request : input.requests) {
        auto request_response = graph.ComputeShortestDistanceFromAtoB(request.from, request.to);
        output.request_responses.emplace_back(request_response);
    }

    return output;
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

Output BruteForceSolve(const Input &input) {
    throw interface::NotImplementedError {};
}

struct TestIo {
    Input input;
    std::optional<Output> optional_expected_output = std::nullopt;

    explicit TestIo(Input input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const interface::NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(Input input, Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

template <
    class Graph, class = std::enable_if_t<std::is_base_of_v<interface::ResizeInterface, Graph>>,
    class =
        std::enable_if_t<std::is_base_of_v<interface::ConnectByNodeIdsInterface<int32_t>, Graph>>>
Graph GenerateRandomGraphWithLoopsAndParallelEdges(int32_t n_nodes, int32_t n_edges) {

    auto node_id_distribution = std::uniform_int_distribution<int32_t>{0, n_nodes - 1};

    Graph graph;
    graph.Resize(n_nodes);

    for (int32_t i = 0; i < n_edges; ++i) {
        auto from = node_id_distribution(*rng::GetEngine());
        auto to = node_id_distribution(*rng::GetEngine());
        graph.Connect(from, to);
    }

    return graph;
}

class RandomGraph : public interface::ResizeInterface,
                    public interface::ConnectByNodeIdsInterface<int32_t> {
public:
    using NodeId = int32_t;
    std::uniform_int_distribution<int32_t> weight_distribution{0, io::kMaxEdgeWeight};

    void Resize(size_t size) override {
    }

    [[nodiscard]] size_t Size() const override {
        return edges.size();
    }

    void Connect(NodeId from, NodeId to) override {
        edges.emplace_back(from, to, weight_distribution(*rng::GetEngine()));
    }

    std::vector<io::DirectedWeightedEdge<io::kMaxEdgeWeight>> edges;
};

std::vector<io::ShortestPathLengthRequest> GenerateRandomRequests(int32_t n_nodes,
                                                                  int32_t n_requests) {
    std::vector<io::ShortestPathLengthRequest> requests;
    requests.reserve(n_requests);

    std::uniform_int_distribution<int32_t> node_id_distribution{0, n_nodes - 1};

    for (int32_t i = 0; i < n_requests; ++i) {
        auto from = node_id_distribution(*rng::GetEngine());
        auto to = node_id_distribution(*rng::GetEngine());
        requests.emplace_back(from, to);
    }

    return requests;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_nodes = 1 + test_case_id;
    int32_t n_edges = 4 * test_case_id;
    int32_t n_requests = 2 * test_case_id;

    auto graph = GenerateRandomGraphWithLoopsAndParallelEdges<RandomGraph>(n_nodes, n_edges);

    Input input;
    input.n_nodes = n_nodes;
    input.edges = graph.edges;
    input.requests = GenerateRandomRequests(n_nodes, n_requests);

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(5'000);
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

int64_t Check(const TestIo &test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    if (test_io.optional_expected_output) {
        auto &expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==================================Expected==================================\n"
               << expected_output
               << "\n==================================Received==================================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    return Check(TestIo{Input{input_stream}, Output{expected}});
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
        "5 4\n"
        "1 2 2\n"
        "2 3 0\n"
        "2 4 2\n"
        "3 4 1\n"
        "2\n"
        "1 4\n"
        "2 5\n",
        "3 -1");

    Check(
        "4 4\n"
        "1 2 0\n"
        "2 3 1\n"
        "3 4 0\n"
        "1 4 2\n"
        "4\n"
        "1 2\n"
        "1 3\n"
        "1 4\n"
        "1 1\n",
        "0 1 1 0");

    Check(
        "4 4\n"
        "1 2 0\n"
        "2 3 1\n"
        "3 4 0\n"
        "1 4 2\n"
        "0\n",
        "");

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 20;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    if (n_random_test_cases > 0) {
        std::cerr << "Random tests OK\n";
    }

    int32_t n_stress_test_cases = 1;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    if (n_stress_test_cases > 0) {
        std::cerr << "Stress tests tests OK\n";
    }

    if (not durations.empty()) {
        auto duration_stats = ComputeStats(durations.begin(), durations.end());
        std::cerr << "Solve duration stats in milliseconds:\n"
                  << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
                  << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
                  << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';
    }

    std::cerr << "OK\n";
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
        std::cout << Solve(Input{std::cin});
    }
    return 0;
}
