// https://contest.yandex.ru/contest/29058/problems/

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return seed;
}

void PrintSeed(std::ostream& ostream = std::cerr) {
    std::cerr << "Seed = " << GetSeed() << std::endl;
}

std::mt19937* GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

namespace hash {
constexpr int32_t kPrime = 1'000'000'007;

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

class SymmetricHash {
public:
    explicit SymmetricHash(int32_t prime = kPrime, std::mt19937* engine = rng::GetEngine())
        : prime_{prime} {
        std::uniform_int_distribution<int32_t> distribution{1, prime_ - 1};
        sum_coefficient_ = distribution(*engine);
        product_coefficient_ = distribution(*engine);
        free_coefficient_ = distribution(*engine);
    }

    int32_t operator()(const std::vector<int32_t>& values) const {
        int64_t sum = 0;
        int64_t product = 1;
        for (const auto& item : values) {
            sum = PositiveMod(sum + item, prime_);
            product = PositiveMod(product * item, prime_);
        }
        return PositiveMod(
            sum_coefficient_ * sum + product_coefficient_ * product + free_coefficient_, prime_);
    }

private:
    int32_t prime_ = 0;
    int64_t sum_coefficient_ = 0;
    int64_t product_coefficient_ = 0;
    int64_t free_coefficient_ = 0;
};

}  // namespace hash

enum class Color { Unvisited, Visited };

struct HashableTreeNode {
    explicit HashableTreeNode(int32_t id) : id{id} {
    }

    Color color = Color::Unvisited;
    int32_t id = 0;
    int32_t second_tallest_child_tree_height = 0;
    int32_t child_tree_hash = 0;
    std::vector<int32_t> neighbor_ids;
};

bool ComparatorNodesByHeightAndHash(const HashableTreeNode& lhv, const HashableTreeNode& rhv) {
    return std::tuple(lhv.second_tallest_child_tree_height, lhv.child_tree_hash) <
           std::tuple(rhv.second_tallest_child_tree_height, rhv.child_tree_hash);
}

bool ComparatorNodesById(const HashableTreeNode& left, const HashableTreeNode& right) {
    return left.id < right.id;
}

bool operator==(const HashableTreeNode& left, const HashableTreeNode& right) {
    return not ComparatorNodesByHeightAndHash(left, right) and
           not ComparatorNodesByHeightAndHash(right, left);
}

class Tree {
public:
    std::vector<HashableTreeNode> nodes;

    Tree() = default;

    explicit Tree(int32_t size) {
        for (int32_t i = 0; i < size; ++i) {
            AddNode();
        }
    }

    void AddNode() {
        nodes.emplace_back(nodes.size());
    }

    void ConnectNodes(int32_t from, int32_t to) {
        nodes[to].neighbor_ids.push_back(from);
        nodes[from].neighbor_ids.push_back(to);
    }

    void Recolor(Color color) {
        for (auto& node : nodes) {
            node.color = color;
        }
    }

    void Recolor(Color color, const std::vector<int32_t>& node_ids) {
        for (auto id : node_ids) {
            nodes[id].color = color;
        }
    }

    [[nodiscard]] std::vector<int32_t> GetNeighborsByColor(int32_t node_id, Color color) const {
        std::vector<int32_t> colored_neighbors;
        for (auto neighbor_id : nodes[node_id].neighbor_ids) {
            if (nodes[neighbor_id].color == color) {
                colored_neighbors.push_back(nodes[neighbor_id].id);
            }
        }
        return colored_neighbors;
    }

    template <class Comparator>
    void Sort(const Comparator& comparator) {
        std::sort(nodes.begin(), nodes.end(), comparator);
    }

    template <class Comparator>
    void SortNodeIds(std::vector<int32_t>* node_ids, const Comparator& comparator) {
        std::sort(node_ids->begin(), node_ids->end(),
                  [this, &comparator](auto lhv, auto rhv) -> bool {
                      return comparator(this->nodes[lhv], this->nodes[rhv]);
                  });
    }

    void ComputeNodeHashes(hash::SymmetricHash hash) {
        Recolor(Color::Unvisited);
        Sort(ComparatorNodesById);

        std::vector<int32_t> unvisited_subtree_degrees;
        std::vector<int32_t> leaves_in_unvisited_subtree;
        for (auto& node : nodes) {
            unvisited_subtree_degrees.push_back(static_cast<int32_t>(node.neighbor_ids.size()));
            if (unvisited_subtree_degrees.back() == 1) {
                leaves_in_unvisited_subtree.push_back(node.id);
            }
        }

        int32_t leaves_second_tallest_child_tree_height = -1;
        while (not leaves_in_unvisited_subtree.empty()) {
            ++leaves_second_tallest_child_tree_height;

            for (auto leaf : leaves_in_unvisited_subtree) {
                nodes[leaf].second_tallest_child_tree_height =
                    leaves_second_tallest_child_tree_height;

                nodes[leaf].child_tree_hash = hash(GetVisitedNeighborsHashes(leaf));
            }

            Recolor(Color::Visited, leaves_in_unvisited_subtree);

            leaves_in_unvisited_subtree = GetNewLeavesInUnvisitedSubtree(
                &unvisited_subtree_degrees, leaves_in_unvisited_subtree);
        }
    }

    [[nodiscard]] int32_t GetRoot() const {
        return std::max_element(nodes.begin(), nodes.end(), ComparatorNodesByHeightAndHash)->id;
    }

private:
    std::vector<int32_t> GetNewLeavesInUnvisitedSubtree(
        std::vector<int32_t>* unvisited_subtree_degrees,
        const std::vector<int32_t>& leaves_in_unvisited_subtree) const {

        std::vector<int32_t> new_leaves_in_unvisited_subtree;
        for (auto leaf : leaves_in_unvisited_subtree) {
            for (auto neighbour : nodes[leaf].neighbor_ids) {
                --(*unvisited_subtree_degrees)[neighbour];
                if (nodes[neighbour].color == Color::Unvisited and
                    (*unvisited_subtree_degrees)[neighbour] == 1) {
                    new_leaves_in_unvisited_subtree.push_back(neighbour);
                }
            }
        }
        return new_leaves_in_unvisited_subtree;
    }

    [[nodiscard]] std::vector<int32_t> GetVisitedNeighborsHashes(int32_t node_id) const {
        std::vector<int32_t> visited_neighbors_hashes;
        for (auto neighbor : GetNeighborsByColor(node_id, Color::Visited)) {
            visited_neighbors_hashes.push_back(nodes[neighbor].child_tree_hash);
        }
        return visited_neighbors_hashes;
    }
};

void RenumerateUnvisitedIsomorphicSubtrees(Tree* first, Tree* second, int32_t first_root,
                                           int32_t second_root,
                                           std::vector<int32_t>* first_to_second_numeration) {
    (*first_to_second_numeration)[first_root] = second_root;

    first->nodes[first_root].color = Color::Visited;
    second->nodes[second_root].color = Color::Visited;

    auto first_unvisited_neighbors = first->GetNeighborsByColor(first_root, Color::Unvisited);
    auto second_unvisited_neighbors = second->GetNeighborsByColor(second_root, Color::Unvisited);

    first->SortNodeIds(&first_unvisited_neighbors, ComparatorNodesByHeightAndHash);
    second->SortNodeIds(&second_unvisited_neighbors, ComparatorNodesByHeightAndHash);

    for (size_t i = 0; i < first_unvisited_neighbors.size(); ++i) {
        RenumerateUnvisitedIsomorphicSubtrees(first, second, first_unvisited_neighbors[i],
                                              second_unvisited_neighbors[i],
                                              first_to_second_numeration);
    }
}

std::vector<int32_t> RenumerateIsomorphicTrees(Tree* first, Tree* second) {
    for (auto tree : {first, second}) {
        tree->Sort(ComparatorNodesById);
        tree->Recolor(Color::Unvisited);
    }

    std::vector<int32_t> first_to_second_numeration(first->nodes.size());
    RenumerateUnvisitedIsomorphicSubtrees(first, second, first->GetRoot(), second->GetRoot(),
                                          &first_to_second_numeration);
    return first_to_second_numeration;
}

class InputType {
public:
    int32_t n_nodes = 0;
    Tree first;
    Tree second;

    explicit InputType(std::istream& in) {
        in >> n_nodes;
        for (int32_t i = 0; i < n_nodes; ++i) {
            first.AddNode();
            second.AddNode();
        }

        for (auto graph : {&first, &second}) {
            for (int32_t i = 0; i < n_nodes - 1; ++i) {
                int32_t from = 0;
                int32_t to = 0;
                in >> from >> to;
                --from;
                --to;
                graph->ConnectNodes(from, to);
            }
        }
    }

    InputType(Tree first, Tree second)
        : n_nodes{static_cast<int32_t>(first.nodes.size())},
          first{std::move(first)},
          second{std::move(second)} {
    }
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(bool are_graphs_isomorphic) : are_graphs_isomorphic{are_graphs_isomorphic} {
    }

    OutputType(bool are_graphs_isomorphic, std::vector<int32_t>&& first_to_second_numeration)
        : are_graphs_isomorphic{are_graphs_isomorphic},
          first_to_second_numeration{std::move(first_to_second_numeration)} {
    }

    explicit OutputType(const std::string& string) {
        if (string != "-1") {
            are_graphs_isomorphic = true;
            std::stringstream ss{string};
            int32_t item = 0;
            while (ss >> item) {
                first_to_second_numeration.push_back(item);
            }
        }
    }

    std::ostream& Write(std::ostream& out) const {
        if (not are_graphs_isomorphic) {
            out << "-1";
        } else {
            for (auto item : first_to_second_numeration) {
                out << item + 1 << '\n';
            }
        }
        return out;
    }

    bool are_graphs_isomorphic = false;
    std::vector<int32_t> first_to_second_numeration;
};

std::ostream& operator<<(std::ostream& os, OutputType const& output) {
    return output.Write(os);
}

OutputType Solve(InputType input) {
    auto& first_tree = input.first;
    auto& second_tree = input.second;

    if (first_tree.nodes.empty()) {
        return OutputType{false};
    }

    hash::SymmetricHash hash;
    for (auto tree : {&first_tree, &second_tree}) {
        tree->ComputeNodeHashes(hash);
        tree->Sort(ComparatorNodesByHeightAndHash);
    }

    if (first_tree.nodes != second_tree.nodes) {
        return OutputType{false};
    }

    return OutputType{true, RenumerateIsomorphicTrees(&first_tree, &second_tree)};
}

namespace test {

std::vector<int32_t> PruferCode(Tree tree) {
    if (tree.nodes.size() < 2) {
        return {};
    }

    tree.Sort(ComparatorNodesById);
    tree.Recolor(Color::Unvisited);

    std::vector<int32_t> code;
    code.reserve(tree.nodes.size() - 2);

    int32_t node_id = 0;
    while (code.size() < tree.nodes.size() - 2) {
        if (tree.nodes[node_id].color == Color::Unvisited and
            tree.nodes[node_id].neighbor_ids.size() == 1) {
            auto neighbor_of_smallest_leaf = tree.nodes[node_id].neighbor_ids.front();
            code.push_back(neighbor_of_smallest_leaf);

            auto& neighbor_ids = tree.nodes[neighbor_of_smallest_leaf].neighbor_ids;
            neighbor_ids.erase(std::find(neighbor_ids.begin(), neighbor_ids.end(), node_id));
            tree.nodes[node_id].color = Color::Visited;

            node_id = std::min(node_id + 1, neighbor_of_smallest_leaf);
        } else {
            ++node_id;
        }
    }
    return code;
}

Tree ReversePruferCode(const std::vector<int32_t>& code) {
    Tree tree(static_cast<int32_t>(code.size() + 2));
    std::vector<int32_t> count(code.size() + 2);
    for (auto number : code) {
        ++count[number];
    }

    int32_t order = 0;
    int32_t next = 0;

    for (auto prufer : code) {
        if (count[next]) {
            while (count[order]) {
                ++order;
            }
            next = order;
        }

        tree.ConnectNodes(next, prufer);

        --count[prufer];
        --count[next];
        if (not count[prufer] and prufer < order) {
            next = prufer;
        }
    }

    while (count[next]) {
        ++next;
    }
    tree.ConnectNodes(next, static_cast<int32_t>(code.size() + 1));

    return tree;
}

Tree GenerateRandomTree(int32_t size, std::mt19937* engine = rng::GetEngine()) {
    if (size < 2) {
        return Tree{size};
    }
    std::uniform_int_distribution<int32_t> nodes_distribution(0, size - 1);
    std::vector<int32_t> prufer_code;
    prufer_code.reserve(size - 2);
    for (int32_t i = 0; i < size - 2; ++i) {
        prufer_code.push_back(nodes_distribution(*engine));
    }
    auto graph = ReversePruferCode(prufer_code);
    return graph;
}

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const& message) : message{message.data()} {
    }

    [[nodiscard]] const char* what() const noexcept override {
        return message;
    }

    const char* message;
};

Tree RenumerateTree(Tree tree, const std::vector<int32_t>& renumeration) {
    for (auto& node : tree.nodes) {
        node.id = renumeration[node.id];
        for (auto& neighbor : node.neighbor_ids) {
            neighbor = renumeration[neighbor];
        }
    }
    std::sort(tree.nodes.begin(), tree.nodes.end(), ComparatorNodesById);
    return tree;
}

void CheckThatRenumerationIsAnIsomorphism(const InputType& input, const OutputType& output) {
    auto first_renumerated = RenumerateTree(input.first, output.first_to_second_numeration);
    if (PruferCode(first_renumerated) != PruferCode(input.second)) {
        throw WrongAnswerException{"False Positive."};
    }
}

void Check(const InputType& input, const OutputType& expected) {
    auto output = Solve(input);
    if (output.are_graphs_isomorphic and not expected.are_graphs_isomorphic) {
        throw WrongAnswerException{"False Positive."};
    } else if (not output.are_graphs_isomorphic and expected.are_graphs_isomorphic) {
        throw WrongAnswerException{"False Negative."};
    } else if (not output.are_graphs_isomorphic and not expected.are_graphs_isomorphic) {
        std::cerr << "Cannot check False Negative, i.e. that graphs are non isomorphic.\n";
    } else {
        CheckThatRenumerationIsAnIsomorphism(input, output);
    }
}

void Check(const std::string& test_case, const std::string& expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    Check(input, OutputType{expected});
}

void Test() {
    rng::PrintSeed();

    {
        Check(
            "4\n"
            "3 1\n"
            "2 3\n"
            "4 3\n"
            "2 4\n"
            "4 1\n"
            "3 4\n",
            "0 1 3 2");
        Check(
            "4\n"
            "2 1\n"
            "2 3\n"
            "4 3\n"
            "3 1\n"
            "1 4\n"
            "2 4\n",
            "1 0 3 2");
        Check("1\n", "0");
        Check(
            "5\n"
            "1 2\n"
            "2 3\n"
            "5 4\n"
            "5 1\n"
            "3 5\n"
            "1 2\n"
            "1 4\n"
            "1 3\n",
            "-1");
        Check(
            "5\n"
            "1 2\n"
            "2 3\n"
            "5 4\n"
            "5 1\n"
            "3 5\n"
            "1 2\n"
            "2 4\n"
            "1 3\n",
            "0 1 3 4 2");
        Check(
            "10\n"
            "1 3\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "6 7\n"
            "6 8\n"
            "8 9\n"
            "8 10\n"

            "1 3\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "5 7\n"
            "6 8\n"
            "8 9\n"
            "8 10\n",
            "-1");
        Check(
            "10\n"
            "1 3\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "6 7\n"
            "6 8\n"
            "8 9\n"
            "8 10\n"

            "1 3\n"
            "2 3\n"
            "3 4\n"
            "4 5\n"
            "5 6\n"
            "6 7\n"
            "6 8\n"
            "8 9\n"
            "8 10\n",
            "0 1 2 3 4 5 6 7 8 9");

        std::stringstream ss{
            "6\n"
            "1 4\n"
            "2 4\n"
            "3 4\n"
            "5 4\n"
            "5 6\n"
            "1 4\n"
            "2 4\n"
            "3 4\n"
            "5 4\n"
            "5 6\n"};
        auto input = InputType{ss};
        std::vector<int32_t> expected{3, 3, 3, 4};
        assert(PruferCode(input.first) == expected);
        assert(ReversePruferCode(expected).nodes == input.first.nodes);
    }

    for (int32_t size = 0; size < 100; ++size) {
        auto tree = GenerateRandomTree(size);
        assert(static_cast<int32_t>(tree.nodes.size()) == size);
        auto code = PruferCode(tree);
        assert(code == PruferCode(ReversePruferCode(code)));
    }

    for (int32_t size_times_hundred = 100; size_times_hundred < 1000; ++size_times_hundred) {
        auto size = size_times_hundred / 100;
        auto first = GenerateRandomTree(size);
        auto second = GenerateRandomTree(size);
        InputType input{first, second};
        auto output = Solve(input);
        if (output.are_graphs_isomorphic) {
            CheckThatRenumerationIsAnIsomorphism(input, output);
            std::cerr << "Random graphs of value " << size << " are isomorphic.\n";
        }
    }

    for (int32_t size_times_ten = 100; size_times_ten < 1000; ++size_times_ten) {
        auto size = size_times_ten / 10;

        auto first = GenerateRandomTree(size);

        std::vector<int32_t> renumeration;
        renumeration.reserve(size);
        for (int32_t i = 0; i < size; ++i) {
            renumeration.push_back(i);
        }
        std::shuffle(renumeration.begin(), renumeration.end(), *rng::GetEngine());

        auto second = RenumerateTree(first, renumeration);

        InputType input{first, second};

        auto output = Solve(input);
        CheckThatRenumerationIsAnIsomorphism(input, output);
        if (renumeration != output.first_to_second_numeration) {
            std::cerr << "Renumerations don't match, but it's not necessarily an error.\n";
        }
    }

    std::cout << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char* argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
