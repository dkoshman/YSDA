#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
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

struct AreNodesConnected {
    struct Request {
        int32_t first = 0;
        int32_t second = 0;
    };

    struct Response {
        bool are_connected = false;
    };
};

struct LinkNodes {
    struct Request {
        int32_t first = 0;
        int32_t second = 0;
    };
};

struct CutNodes {
    struct Request {
        int32_t first = 0;
        int32_t second = 0;
    };
};

using Request = std::variant<AreNodesConnected::Request, LinkNodes::Request, CutNodes::Request>;

class Input {
public:
    int32_t n_nodes = 0;
    std::vector<Request> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_requests = 0;
        in >> n_nodes >> n_requests;
        requests.reserve(n_requests);

        std::string request_type;
        int32_t first = 0;
        int32_t second = 0;
        while (in >> request_type >> first >> second) {
            --first;
            --second;
            if (request_type == "get") {
                requests.emplace_back(AreNodesConnected::Request{first, second});
            } else if (request_type == "cut") {
                requests.emplace_back(CutNodes::Request{first, second});
            } else if (request_type == "link") {
                requests.emplace_back(LinkNodes::Request{first, second});
            } else {
                throw std::invalid_argument{"Unknown request type."};
            }
        }
    }
};

class Output {
public:
    std::vector<bool> are_nodes_connected;

    Output() = default;

    explicit Output(const std::vector<AreNodesConnected::Response> &responses) {
        for (auto response : responses) {
            are_nodes_connected.emplace_back(response.are_connected);
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            are_nodes_connected.emplace_back(item == 1);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : are_nodes_connected) {
            out << (item ? 1 : -1) << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return are_nodes_connected != other.are_nodes_connected;
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

class Treap {
public:
    template <class T>
    void SetLeftChild(T new_left_child) {
        throw NotImplementedError{};
    }

    template <class Treap>
    void SetRightChild(Treap new_right_child) {
        throw NotImplementedError{};
    }

    template <class Treap>
    Treap DropLeftChild() {
        throw NotImplementedError{};
    }

    template <class Treap>
    Treap DropRightChild() {
        throw NotImplementedError{};
    }

    template <class Treap>
    [[nodiscard]] bool CompareKey(const Treap &other) const {
        throw NotImplementedError{};
    }

    template <class Treap>
    [[nodiscard]] bool ComparePriority(const Treap &other) const {
        throw NotImplementedError{};
    }
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

template <class Treap>
bool ComparatorTreapKey(const Treap &left, const Treap &right) {
    return left.CompareKey(right);
}

template <class TreapPtr>
bool ComparatorTreapPtrKey(const TreapPtr &left, const TreapPtr &right) {
    return left->CompareKey(*right);
}

template <class Treap>
bool ComparatorTreapPriority(const Treap &left, const Treap &right) {
    return left.ComparePriority(right);
}

template <class TreapPtr>
bool ComparatorTreapPtrPriority(const TreapPtr &left, const TreapPtr &right) {
    return left->ComparePriority(*right);
}

template <class NodePtrIterator, class NodeComparator>
typename NodePtrIterator::value_type ConstructCartesianTree(NodePtrIterator begin,
                                                            NodePtrIterator end,
                                                            const NodeComparator &comparator) {

    if (begin == end) {
        return {};
    }

    auto min_iter = std::min_element(begin, end, comparator);
    auto min_node = *min_iter;

    min_node->SetLeftChild(ConstructCartesianTree(begin, min_iter, comparator));

    min_node->SetRightChild(ConstructCartesianTree(min_iter + 1, end, comparator));

    return min_node;
}

template <class TreapPtrIterator>
typename TreapPtrIterator::value_type ConstructTreap(TreapPtrIterator begin, TreapPtrIterator end) {

    std::sort(begin, end, ComparatorTreapPtrKey<typename TreapPtrIterator::value_type>);

    return ConstructCartesianTree(
        begin, end, ComparatorTreapPtrPriority<typename TreapPtrIterator::value_type>);
}

class NodeAlreadyHasParentError : public std::logic_error {
public:
    NodeAlreadyHasParentError() : std::logic_error("Node already has a parent."){};
};

template <class TreapPtr>
void SetLeftChild(const TreapPtr &treap, TreapPtr new_left) {
    if (new_left and new_left->parent) {
        throw NodeAlreadyHasParentError{};
    }
    treap->DropLeftChild();
    treap->left = std::move(new_left);
    if (treap->left) {
        treap->left->parent = treap;
    }
}

template <class TreapPtr>
void SetRightChild(const TreapPtr &treap, TreapPtr new_right) {
    if (new_right and new_right->parent) {
        throw NodeAlreadyHasParentError{};
    }
    treap->DropRightChild();
    treap->right = std::move(new_right);
    if (treap->right) {
        treap->right->parent = treap;
    }
}

template <class TreapPtr>
TreapPtr Merge(TreapPtr left, TreapPtr right) {
    if (not left) {
        return right;
    }

    if (not right) {
        return left;
    }

    if (left->ComparePriority(*right)) {

        auto right_child = left->DropRightChild();
        SetRightChild(left, Merge(right_child, right));
        return left;
    } else {

        auto left_child = right->DropLeftChild();
        SetLeftChild(right, Merge(left, left_child));
        return right;
    }
}

template <class TreapPtr>
TreapPtr Merge(std::initializer_list<TreapPtr> list) {
    auto treap = *list.begin();
    for (auto it = list.begin() + 1; it != list.end(); ++it) {
        treap = Merge(treap, *it);
    }
    return treap;
}

template <class Value>
struct Pair {
    Value left;
    Value right;
};

template <class TreapPtr>
Pair<TreapPtr> Split(TreapPtr treap, int32_t key_value) {
    if (not treap) {
        return {};
    }

    if (treap->CompareKey(key_value)) {

        auto right_child = treap->DropRightChild();
        auto right_child_split = Split(right_child, key_value);
        treap->SetRightChild(right_child_split.left);
        return {treap, right_child_split.right};
    } else {

        auto left_child = treap->DropLeftChild();
        auto left_child_split_by_index = Split(left_child, key_value);
        treap->SetLeftChild(left_child_split_by_index.right);
        return {left_child_split_by_index.left, treap};
    }
}

}  // namespace implementation

class DirectedEdge : public implementation::DirectedEdge {
public:
    using implementation::DirectedEdge::DirectedEdge;

    bool operator==(const DirectedEdge &other) const {
        return from == other.from and to == other.to;
    }

    struct HashFunction {
        std::size_t operator()(const DirectedEdge &edge) const {
            return edge.from ^ (edge.to + 0x9e3779b9 + (edge.to << 6) + (edge.to >> 2));
        }
    };
};

class CartesianTreap {
public:
    DirectedEdge edge;
    int32_t priority = 0;
    CartesianTreap *left = nullptr;
    CartesianTreap *right = nullptr;
    CartesianTreap *parent = nullptr;

    explicit CartesianTreap(DirectedEdge edge)
        : edge{edge}, priority{static_cast<int32_t>(rng::GetEngine()->operator()())} {
    }

    [[nodiscard]] bool ComparePriority(const CartesianTreap &other) const {
        return priority < other.priority;
    }

    [[nodiscard]] CartesianTreap *GetRoot() {
        return parent ? parent->GetRoot() : this;
    }

    CartesianTreap *DropLeftChild() {
        auto left_child = left;
        if (left_child) {
            left_child->parent = nullptr;
        }
        left = nullptr;
        return left_child;
    }

    CartesianTreap *DropRightChild() {
        auto right_child = right;
        if (right_child) {
            right_child->parent = nullptr;
        }
        right = nullptr;
        return right_child;
    }
};

implementation::Pair<CartesianTreap *> SplitAtNode(CartesianTreap *treap, bool keep_node = true) {
    auto left = treap->DropLeftChild();
    auto right = keep_node ? treap : treap->DropRightChild();

    while (treap->parent) {
        auto parent = treap->parent;
        if (treap == parent->left) {
            parent->DropLeftChild();
            implementation::SetLeftChild(parent, right);
            right = parent;
        } else {
            parent->DropRightChild();
            implementation::SetRightChild(parent, left);
            left = parent;
        }
        treap = parent;
    }

    return {left, right};
}

class ForestConnectivity {
public:
    std::unordered_map<DirectedEdge, std::unique_ptr<CartesianTreap>, DirectedEdge::HashFunction>
        euler_edge_to_treap_map;

    explicit ForestConnectivity(int32_t n_nodes) {
        for (int32_t node = 0; node < n_nodes; ++node) {
            AddEulerNode(node);
        }
    }

    [[nodiscard]] bool AreConnected(int32_t first, int32_t second) {
        return GetTreapRootByEulerNode(first) == GetTreapRootByEulerNode(second);
    }

    void Link(int32_t first, int32_t second) {
        MoveEulerNodeToFront(first);
        MoveEulerNodeToFront(second);

        auto first_to_second = AddEulerEdge(first, second);
        auto second_to_first = AddEulerEdge(second, first);

        auto first_treap = GetTreapRootByEulerNode(first);
        auto second_treap = GetTreapRootByEulerNode(second);

        implementation::Merge({first_treap, first_to_second, second_treap, second_to_first});
    }

    void Cut(int32_t first, int32_t second) {
        MoveEulerNodeToFront(first);

        auto first_to_second = euler_edge_to_treap_map[DirectedEdge{first, second}].get();
        auto [first_root, second_then_first_right] = SplitAtNode(first_to_second, false);

        auto second_to_first = euler_edge_to_treap_map[DirectedEdge{second, first}].get();
        auto [second_treap, first_right] = SplitAtNode(second_to_first, false);

        implementation::Merge(first_root, GetTreapRoot(first_right));
    }

private:
    CartesianTreap *GetTreapRootByEulerNode(int32_t node) {
        return GetTreapRoot(GetTreapNode(node));
    }

    CartesianTreap *GetTreapNode(int32_t node) {
        return euler_edge_to_treap_map[DirectedEdge{node, node}].get();
    }

    CartesianTreap *GetTreapRoot(CartesianTreap *treap_node) {
        return treap_node ? euler_edge_to_treap_map[treap_node->GetRoot()->edge].get() : treap_node;
    }

    CartesianTreap *AddEulerEdge(int32_t from, int32_t to) {
        DirectedEdge edge{from, to};
        euler_edge_to_treap_map.erase(edge);
        euler_edge_to_treap_map[edge] = std::make_unique<CartesianTreap>(edge);
        return euler_edge_to_treap_map[edge].get();
    }

    CartesianTreap *AddEulerNode(int32_t node) {
        return AddEulerEdge(node, node);
    }

    void MoveEulerNodeToFront(int32_t node) {
        auto [left, right] = SplitAtNode(GetTreapNode(node));
        implementation::Merge(right, left);
    }
};

io::Output Solve(const io::Input &input) {

    ForestConnectivity forest_connectivity(input.n_nodes);
    io::Output output;

    for (auto &request : input.requests) {
        if (auto are_connected_request = std::get_if<io::AreNodesConnected::Request>(&request)) {
            output.are_nodes_connected.emplace_back(forest_connectivity.AreConnected(
                are_connected_request->first, are_connected_request->second));
        } else if (auto link_request = std::get_if<io::LinkNodes::Request>(&request)) {
            forest_connectivity.Link(link_request->first, link_request->second);
        } else if (auto cut_request = std::get_if<io::CutNodes::Request>(&request)) {
            forest_connectivity.Cut(cut_request->first, cut_request->second);
        } else {
            throw std::invalid_argument{"Unknown request."};
        }
    }

    return output;
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

io::Output BruteForceSolve(const io::Input &input) {
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_nodes = 1 + test_case_id;
    int32_t n_requests = 1 + test_case_id;

    std::uniform_int_distribution<int32_t> distribution{0, n_nodes - 1};
    io::Input input;
    input.n_nodes = n_nodes;
    ForestConnectivity forest_connectivity(input.n_nodes);
    std::vector<io::LinkNodes::Request> links;

    while (input.requests.size() < static_cast<size_t>(n_requests)) {
        auto request_id = distribution(*rng::GetEngine());
        auto first = distribution(*rng::GetEngine());
        auto second = distribution(*rng::GetEngine());

        input.requests.emplace_back(io::AreNodesConnected::Request{first, second});

        if (static_cast<int32_t>(links.size()) < n_nodes + request_id) {
            for (second = 0; second < n_nodes; ++second) {
                if (not forest_connectivity.AreConnected(first, second)) {
                    auto link = io::LinkNodes::Request{first, second};
                    input.requests.emplace_back(link);
                    links.emplace_back(link);
                    forest_connectivity.Link(first, second);
                    break;
                }
            }
        } else if (not links.empty()) {
            std::shuffle(links.begin(), links.end(), *rng::GetEngine());
            input.requests.emplace_back(
                io::CutNodes::Request{links.back().first, links.back().second});
            forest_connectivity.Cut(links.back().first, links.back().second);
            links.pop_back();
        }
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(10'000);
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

    TimedChecker timed_checker;

    timed_checker.Check(
        "3 7\n"
        "get 1 2\n"
        "link 1 2\n"
        "get 1 2\n"
        "cut 1 2\n"
        "get 1 2\n"
        "link 1 2\n"
        "get 1 2",
        "-1\n"
        "1\n"
        "-1\n"
        "1");

    timed_checker.Check(
        "5 10\n"
        "link 1 2\n"
        "link 2 3\n"
        "link 4 3\n"
        "cut 3 4\n"
        "get 1 2\n"
        "get 1 3\n"
        "get 1 4\n"
        "get 2 3\n"
        "get 2 4\n"
        "get 3 4",
        "1\n"
        "1\n"
        "-1\n"
        "1\n"
        "-1\n"
        "-1");

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
