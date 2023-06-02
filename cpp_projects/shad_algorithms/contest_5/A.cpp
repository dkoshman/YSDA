// https://contest.yandex.ru/contest/29060/problems/

#include <algorithm>
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

enum class NodeVisitedState { Unvisited, Visited };

struct Node {
    int32_t id = 0;
    NodeVisitedState state = NodeVisitedState::Unvisited;
    std::optional<int32_t> dfs_exit_order = std::nullopt;

    explicit Node(int32_t id) : id{id} {
    }
};

bool ComparatorNodesByDfsExitOrder(const Node &left, const Node &right) {
    return left.dfs_exit_order < right.dfs_exit_order;
}

class Graph {
public:
    using NodeFunction = std::function<void(Node &)>;
    using AdjacentIds = std::vector<std::vector<int32_t>>;

    struct DfsArguments {
        int32_t node_id_to_start_from = 0;
        const AdjacentIds *adjacent_ids = nullptr;
        std::optional<NodeFunction> on_enter = std::nullopt;
        std::optional<NodeFunction> on_exit = std::nullopt;
    };

    Graph() = default;

    explicit Graph(size_t size) {
        Resize(size);
    }

    void Resize(size_t size) {
        nodes.reserve(size);
        while (nodes.size() < size) {
            AddNode();
        }
        while (nodes.size() > size) {
            nodes.pop_back();
        }
    }

    void AddNode() {
        nodes.emplace_back(nodes.size());
        adjacent_ids.resize(nodes.size());
        reverse_adjacent_ids.resize(nodes.size());
    }

    void ConnectNodes(int32_t from, int32_t to) {
        adjacent_ids[from].push_back(to);
        reverse_adjacent_ids[to].push_back(from);
    }

    Node &operator[](int32_t node_id) {
        return nodes[node_id];
    }

    Node operator[](int32_t node_id) const {
        return nodes[node_id];
    }

    void ApplyToNodes(const NodeFunction &function,
                      std::optional<std::vector<int32_t>> node_ids = std::nullopt) {
        if (node_ids) {
            for (int32_t node_id : node_ids.value()) {
                function(nodes[node_id]);
            }
        } else {
            for (auto &node : nodes) {
                function(node);
            }
        }
    }

    void ResetNodesState(NodeVisitedState state = NodeVisitedState::Unvisited) {
        ApplyToNodes([state](Node &node) { node.state = state; });
    }

    void ResetNodesFinishOrder() {
        ApplyToNodes([](Node &node) { node.dfs_exit_order = std::nullopt; });
    }

    void DepthFirstSearch(DfsArguments arguments) {
        if (not arguments.adjacent_ids) {
            arguments.adjacent_ids = &adjacent_ids;
        }
        DepthFirstSearchRecursive(arguments, arguments.node_id_to_start_from);
    }

    void AssignDfsFinishOrderToAllNodes() {
        ResetNodesState();
        ResetNodesFinishOrder();

        auto fictive_root_id = AddFictiveRoot();

        DfsArguments arguments{fictive_root_id};

        int32_t dfs_exit_order = 0;
        arguments.on_exit = [&dfs_exit_order](Node &node) {
            node.dfs_exit_order = dfs_exit_order++;
        };

        DepthFirstSearch(arguments);

        RemoveFictiveRoot();
    }

    int32_t FindSizeOfSmallestStronglyConnectedComponentWithoutEdgesLeadingOutFromIt() {
        AssignDfsFinishOrderToAllNodes();

        ResetNodesState();

        auto nodes_by_exit_order = nodes;
        std::sort(nodes_by_exit_order.begin(), nodes_by_exit_order.end(),
                  ComparatorNodesByDfsExitOrder);

        auto smallest_super_scc_size = static_cast<int32_t>(nodes.size());
        for (auto node : nodes_by_exit_order) {
            if (nodes[node.id].state == NodeVisitedState::Unvisited) {

                auto super_scc_size =
                    DfsThroughLeafStronglyConnectedComponentWithoutATraceAndReturnItsSize(node.id);

                smallest_super_scc_size = std::min(smallest_super_scc_size, super_scc_size);

                SimpleReverseDfs(node.id);
            }
        }

        return smallest_super_scc_size;
    }

    std::vector<Node> nodes;
    AdjacentIds adjacent_ids;
    AdjacentIds reverse_adjacent_ids;

private:
    void DepthFirstSearchRecursive(const DfsArguments &arguments, int32_t node_id) {
        auto &node = nodes[node_id];

        node.state = NodeVisitedState::Visited;

        if (arguments.on_enter) {
            arguments.on_enter.value()(node);
        }

        for (auto adjacent_id : (*arguments.adjacent_ids)[node_id]) {
            if (nodes[adjacent_id].state == NodeVisitedState::Unvisited) {
                DepthFirstSearchRecursive(arguments, adjacent_id);
            }
        }

        if (arguments.on_exit) {
            arguments.on_exit.value()(node);
        }
    }

    int32_t AddFictiveRoot() {
        AddNode();
        auto fictive_root_id = nodes.back().id;

        adjacent_ids[fictive_root_id].reserve(fictive_root_id);
        for (int32_t i = 0; i < fictive_root_id; ++i) {
            adjacent_ids[fictive_root_id].push_back(i);
        }
        return fictive_root_id;
    }

    void RemoveFictiveRoot() {
        adjacent_ids.resize(nodes.back().id);
        reverse_adjacent_ids.resize(nodes.back().id);
        nodes.pop_back();
    }

    int32_t DfsThroughLeafStronglyConnectedComponentWithoutATraceAndReturnItsSize(
        int32_t node_id_to_start_from) {

        DfsArguments arguments{node_id_to_start_from};

        std::vector<int32_t> visited_node_ids;
        int32_t super_scc_size = 0;
        arguments.on_exit = [&super_scc_size, &visited_node_ids](Node &node) {
            ++super_scc_size;
            visited_node_ids.push_back(node.id);
        };

        DepthFirstSearch(arguments);

        for (auto node_id : visited_node_ids) {
            nodes[node_id].state = NodeVisitedState::Unvisited;
        }

        return super_scc_size;
    }

    void SimpleReverseDfs(int32_t node_id_to_start_from) {
        DfsArguments reverse_search_arguments{node_id_to_start_from};
        reverse_search_arguments.adjacent_ids = &reverse_adjacent_ids;
        DepthFirstSearch(reverse_search_arguments);
    }
};

namespace io {

class Input {
public:
    enum class GameResult {
        first_won = 1,
        second_won = 2,
        draw = 3,
    };

    struct GameLog {
        int32_t first_player_id_plus_one = 0;
        int32_t second_player_id_plus_one = 0;
        GameResult game_result = GameResult::draw;

        GameLog() = default;

        GameLog(int32_t first_player_id_plus_one, int32_t second_player_id_plus_one,
                int32_t game_result_id)
            : first_player_id_plus_one{first_player_id_plus_one},
              second_player_id_plus_one{second_player_id_plus_one},
              game_result{game_result_id} {
        }
    };

    int32_t n_players = 0;
    std::vector<GameLog> game_logs;

    Input() = default;

    explicit Input(std::istream &in) {
        size_t n_games = 0;
        in >> n_players >> n_games;
        game_logs.reserve(n_players);

        for (size_t index = 0; index < n_games; ++index) {
            int32_t first = 0;
            int32_t second = 0;
            int32_t game_result_id = 0;

            in >> first >> second >> game_result_id;
            game_logs.emplace_back(first, second, game_result_id);
        }
    }

    [[nodiscard]] Graph GetGraph() const {
        Graph graph;
        graph.Resize(n_players);

        for (auto game_log : game_logs) {
            auto first = game_log.first_player_id_plus_one - 1;
            auto second = game_log.second_player_id_plus_one - 1;

            if (game_log.game_result == GameResult::first_won) {
                graph.ConnectNodes(second, first);
            } else if (game_log.game_result == GameResult::second_won) {
                graph.ConnectNodes(first, second);
            }
        }

        return graph;
    };
};

class Output {
public:
    int32_t request_responses = 0;

    Output() = default;

    explicit Output(int32_t size_of_biggest_cohesive_company)
        : request_responses{size_of_biggest_cohesive_company} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << request_responses << '\n';
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

Output Solve(const Input &input) {
    auto graph = input.GetGraph();

    auto smallest_super_scc_size =
        graph.FindSizeOfSmallestStronglyConnectedComponentWithoutEdgesLeadingOutFromIt();

    return Output{input.n_players - smallest_super_scc_size + 1};
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

struct Vertex {
    enum Color { White, Grey };
    Color color = Color::White;
    std::vector<int32_t> edges;
    std::vector<int32_t> reverse_edges;
};

using Graph = std::vector<Vertex>;

struct VertexOrder {
    int32_t vertex_index = 0;
    int32_t order = 0;
};

void DepthFirstTraversal(Graph &graph, std::vector<VertexOrder> &orders, int32_t vertex_index,
                         int32_t &order) {

    auto &vertex = graph[vertex_index];
    vertex.color = Vertex::Color::Grey;

    for (auto edge : vertex.edges) {
        if (graph[edge].color == Vertex::Color::White) {
            DepthFirstTraversal(graph, orders, edge, order);
        }
    }
    orders.push_back({vertex_index, ++order});
}

void TraverseLeafComponent(Graph &graph, int32_t vertex_index, int32_t &component_size) {
    auto &vertex = graph[vertex_index];
    vertex.color = Vertex::Color::White;
    for (auto edge : vertex.edges) {
        if (graph[edge].color == Vertex::Color::Grey) {
            TraverseLeafComponent(graph, edge, ++component_size);
        }
    }
}

void ClearColors(Graph &graph, int32_t vertex_index) {
    auto &vertex = graph[vertex_index];
    vertex.color = Vertex::Color::Grey;
    for (auto edge : vertex.edges) {
        if (graph[edge].color == Vertex::Color::White) {
            ClearColors(graph, edge);
        }
    }
}

void ReverseTraversal(Graph &graph, int32_t vertex_index) {
    auto &vertex = graph[vertex_index];
    vertex.color = Vertex::Color::White;
    for (auto edge : vertex.reverse_edges) {
        if (graph[edge].color == Vertex::Color::Grey) {
            ReverseTraversal(graph, edge);
        }
    }
}

std::vector<int32_t> CalculateLeafComponentSizes(Graph &graph, std::vector<VertexOrder> &orders) {
    std::vector<int32_t> result;
    for (auto order : orders) {
        if (graph[order.vertex_index].color == Vertex::Color::Grey) {
            int32_t component_size = 1;
            TraverseLeafComponent(graph, order.vertex_index, component_size);
            result.push_back(component_size);
            ClearColors(graph, order.vertex_index);
            ReverseTraversal(graph, order.vertex_index);
        }
    }
    return result;
}

int32_t FindTheSizeOfBiggestCohesiveCompany(Graph graph) {
    auto vertices_count = static_cast<int32_t>(graph.size());
    std::vector<int32_t> fake_edges;
    fake_edges.reserve(vertices_count);
    for (int32_t i = 0; i < vertices_count; ++i) {
        fake_edges.push_back(i);
    }
    graph.push_back({Vertex::Color::White, fake_edges, {}});

    std::vector<VertexOrder> orders;
    int32_t order = 0;
    DepthFirstTraversal(graph, orders, vertices_count, order);
    graph.pop_back();
    std::sort(orders.begin(), orders.end(),
              [](const VertexOrder &lhv, const VertexOrder &rhv) { return lhv.order < rhv.order; });
    orders.pop_back();
    auto component_sizes = CalculateLeafComponentSizes(graph, orders);
    auto min_component_size = *std::min_element(component_sizes.begin(), component_sizes.end());
    return vertices_count - min_component_size + 1;
}

Output BruteForceSolve(const Input &input) {
    auto my_graph = input.GetGraph();
    Graph graph(my_graph.nodes.size());
    for (auto node : my_graph.nodes) {
        for (auto i : my_graph.adjacent_ids[node.id]) {
            graph[node.id].edges.push_back(i);
            graph[i].reverse_edges.push_back(node.id);
        }
    }
    return Output{FindTheSizeOfBiggestCohesiveCompany(graph)};
}

struct TestIo {
    Input input;
    std::optional<Output> optional_expected_output = std::nullopt;

    explicit TestIo(Input input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(Input input, Output output) : input{std::move(input)}, optional_expected_output{output} {
    }
};

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    //    int32_t n_players = 2 + test_case_id * 2;
    //    int32_t n_games = 1 + test_case_id * 10;
    int32_t n_players = 5;
    int32_t n_games = 10;

    auto player_distribution = std::uniform_int_distribution<>{1, n_players};
    auto game_result_id_distribution = std::uniform_int_distribution<>{1, 3};

    Input input;
    input.n_players = n_players;

    while (static_cast<int32_t>(input.game_logs.size()) < n_games) {
        auto first = player_distribution(*rng::GetEngine());
        auto second = player_distribution(*rng::GetEngine());
        if (first == second) {
            continue;
        }
        auto game_result_id = game_result_id_distribution(*rng::GetEngine());
        input.game_logs.emplace_back(first, second, game_result_id);
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(50'000);
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

int64_t Check(const std::string &test_case, int32_t expected) {
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
        "2 1\n"
        "1 2 3\n",
        2);
    Check(
        "2 3\n"
        "1 2 1\n"
        "1 2 2\n"
        "1 2 3\n",
        1);
    Check(
        "3 2\n"
        "2 3 1\n"
        "2 3 2\n",
        3);
    Check(
        "3 3\n"
        "2 3 1\n"
        "2 3 2\n"
        "2 1 1\n",
        2);
    Check(
        "3 3\n"
        "1 2 1\n"
        "1 2 2\n"
        "2 3 1\n",
        2);
    Check(
        "4 3\n"
        "1 2 1\n"
        "2 3 1\n"
        "3 4 1\n",
        4);
    Check(
        "3 6\n"
        "1 2 1\n"
        "1 2 2\n"
        "1 3 1\n"
        "1 3 2\n"
        "2 3 1\n"
        "2 3 2\n",
        1);
    Check(
        "7 8\n"
        "1 7 1\n"
        "2 7 1\n"
        "4 7 1\n"
        "2 3 1\n"
        "3 2 1\n"
        "4 5 1\n"
        "6 5 2\n"
        "4 6 2\n",
        7);
    Check(
        "9 19\n"
        "1 6 1\n"
        "1 7 1\n"
        "2 6 1\n"
        "2 7 1\n"
        "1 2 1\n"
        "1 2 2\n"
        "3 6 1\n"
        "3 7 1\n"
        "4 6 1\n"
        "4 7 1\n"
        "5 6 1\n"
        "5 7 1\n"
        "3 4 1\n"
        "4 5 1\n"
        "5 3 1\n"
        "6 8 1\n"
        "6 9 1\n"
        "7 8 1\n"
        "7 9 1\n",
        8);

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 100;

    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    std::cerr << "Random tests OK\n";

    int32_t n_stress_test_cases = 1;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    std::cerr << "Stress tests tests OK\n";

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

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
