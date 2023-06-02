#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

constexpr int32_t kPrime = 10'007;

namespace rng {

uint32_t GetSeed() {
    static std::random_device random_device;
    auto seed = random_device();
    return seed;
}

std::mt19937& GetEngine() {
    static std::mt19937 engine(GetSeed());
    return engine;
}

}  // namespace rng

struct AllocateRequest {
    int32_t memory_size = 0;
};

struct DeallocateRequest {
    int32_t request_id_of_allocate_request = 0;

    explicit DeallocateRequest(int32_t request_id) : request_id_of_allocate_request{request_id} {
    }
};

using Request = std::variant<AllocateRequest, DeallocateRequest>;

class InputType {
public:
    std::vector<Request> requests;
    int32_t total_memory_size = 0;

    explicit InputType(int32_t total_memory_size) : total_memory_size{total_memory_size} {
    }

    explicit InputType(std::istream& in) {
        int32_t request_count = 0;
        in >> total_memory_size >> request_count;
        requests.reserve(request_count);

        for (int32_t index = 0; index < request_count; ++index) {
            int32_t value = 0;
            in >> value;

            if (value > 0) {
                requests.emplace_back(AllocateRequest{value});
            } else if (value < 0) {
                requests.emplace_back(DeallocateRequest{-value - 1});
            } else {
                throw std::invalid_argument{"Unknown value " + std::to_string(value) + '\n'};
            }
        }
    }
};

struct IndexedNumber {
    int32_t begin_position;
    int32_t value;

    [[nodiscard]] int32_t GetEndPosition() const {
        return begin_position + value;
    }

    bool operator==(const IndexedNumber& other) const {
        return std::tie(begin_position, value) == std::tie(other.begin_position, other.value);
    }
};

struct RegisteredMemoryBlock : public IndexedNumber {
    int32_t id;

    RegisteredMemoryBlock(int32_t id, IndexedNumber memory_block)
        : IndexedNumber{memory_block}, id{id} {
    }

    [[nodiscard]] IndexedNumber GetMemoryBlock() const {
        return IndexedNumber{begin_position, value};
    }
};

struct DeclinedAllocateResponse {
    bool operator==(const DeclinedAllocateResponse& other) const {
        return true;
    }
};

struct ApprovedAllocateResponse {
    RegisteredMemoryBlock allocated_memory_block;

    bool operator==(const ApprovedAllocateResponse& other) const {
        return allocated_memory_block == other.allocated_memory_block;
    }
};

using AllocateResponse = std::variant<DeclinedAllocateResponse, ApprovedAllocateResponse>;

struct IgnoredDeallocateResponse {};

struct ApprovedDeallocateResponse {};

using DeallocateResponse = std::variant<IgnoredDeallocateResponse, ApprovedDeallocateResponse>;

using Response = std::variant<AllocateResponse, DeallocateResponse>;

class OutputType {
public:
    std::vector<int32_t> allocated_memory_blocks_begin_position;

    OutputType() = default;

    //    explicit OutputType(std::vector<int32_t>&& allocated_memory_blocks_begin_position)
    //        : allocated_memory_blocks_begin_position{allocated_memory_blocks_begin_position} {
    //    }

    explicit OutputType(const std::vector<Response>& responses) {
        allocated_memory_blocks_begin_position.reserve(responses.size());
        for (const auto& response : responses) {
            if (auto allocate_response = std::get_if<AllocateResponse>(&response)) {
                if (std::holds_alternative<DeclinedAllocateResponse>(*allocate_response)) {
                    allocated_memory_blocks_begin_position.push_back(-1);
                } else {
                    allocated_memory_blocks_begin_position.push_back(
                        std::get<ApprovedAllocateResponse>(*allocate_response)
                            .allocated_memory_block.begin_position);
                }
            }
        }
    }

    explicit OutputType(const std::string& string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            allocated_memory_blocks_begin_position.push_back(item);
        }
    }

    std::ostream& Write(std::ostream& out) const {
        for (auto item : allocated_memory_blocks_begin_position) {
            if (item >= 0) {
                ++item;
            }
            out << item << '\n';
        }
        return out;
    }

    bool operator==(const OutputType& other) const {
        return allocated_memory_blocks_begin_position ==
               other.allocated_memory_blocks_begin_position;
    }
};

std::ostream& operator<<(std::ostream& os, OutputType const& output) {
    return output.Write(os);
}

bool ComparatorMemorySizePositionGreater(const RegisteredMemoryBlock& lhv,
                                         const RegisteredMemoryBlock& rhv) {
    return std::tie(lhv.value, rhv.begin_position) > std::tie(rhv.value, lhv.begin_position);
}

template <class ValueType>
class Heap {
public:
    using Comparator = std::function<bool(const ValueType& lhv, const ValueType&)>;

    Heap(size_t capacity, Comparator comparator) : comparator_{std::move(comparator)} {
        nodes_.reserve(capacity);
    }

    [[nodiscard]] size_t Size() const {
        return nodes_.size();
    }

    [[nodiscard]] bool Empty() const {
        return nodes_.empty();
    }

    [[nodiscard]] ValueType Top() const {
        return nodes_.front();
    }

    void Insert(ValueType block) {
        nodes_.push_back(block);
        SiftUp(static_cast<int32_t>(nodes_.size()) - 1);
    }

    void Pop() {
        std::iter_swap(nodes_.begin(), nodes_.end() - 1);
        nodes_.pop_back();
        SiftDown(0);
    }

private:
    [[nodiscard]] bool CompareByIndex(int32_t left_node, int32_t right_node) const {
        return comparator_(nodes_[left_node], nodes_[right_node]);
    }

    void SiftUp(int32_t node) {
        while (node != 0 and CompareByIndex(node, Parent(node))) {
            SwapByNodeIndex(node, Parent(node));
            node = Parent(node);
        }
    }

    void SiftDown(int32_t node) {
        int32_t left = Left(node);
        int32_t right = Right(node);
        auto size = static_cast<int32_t>(nodes_.size());

        while ((left < size and CompareByIndex(left, node)) or
               (right < size and CompareByIndex(right, node))) {

            if (right >= size or CompareByIndex(left, right)) {
                SwapByNodeIndex(node, left);
                node = left;
            } else {
                SwapByNodeIndex(node, right);
                node = right;
            }
            left = Left(node);
            right = Right(node);
        }
    }

    [[nodiscard]] static int32_t Left(int32_t node) {
        return node * 2 + 1;
    }

    [[nodiscard]] static int32_t Right(int32_t node) {
        return node * 2 + 2;
    }

    [[nodiscard]] static int32_t Parent(int32_t node) {
        return (node - 1) / 2;
    }

    void SwapByNodeIndex(int32_t left_node, int32_t right_node) {
        std::iter_swap(nodes_.begin() + left_node, nodes_.begin() + right_node);
    }

    std::vector<ValueType> nodes_;
    Comparator comparator_;
};

class Hash {
public:
    explicit Hash(int32_t prime_image_size) : prime_{prime_image_size} {
        std::uniform_int_distribution<int32_t> distribution{1, prime_image_size - 1};
        coefficient_ = distribution(rng::GetEngine());
    }

    [[nodiscard]] int32_t operator()(int64_t value) const {
        return static_cast<int32_t>(value * coefficient_ % prime_);
    }

private:
    int64_t prime_ = 0;
    int32_t coefficient_ = 0;
};

class HashSet {
public:
    explicit HashSet(int32_t prime_table_size) : hash_{prime_table_size}, table_(prime_table_size) {
    }

    void Insert(int32_t key, int32_t value) {
        table_[hash_(key)].push_back(value);
    }

    [[nodiscard]] std::vector<int32_t> GetPreimage(int32_t key) const {
        return table_[hash_(key)];
    }

private:
    Hash hash_;
    std::vector<std::vector<int32_t>> table_;
};

class MemoryBlockDispenser {
public:
    explicit MemoryBlockDispenser(int32_t total_memory_size, int32_t prime_hash_table_size = kPrime)
        : block_begin_to_id_table_(prime_hash_table_size),
          block_end_to_id_table_(prime_hash_table_size) {
        RegisterNewFreeMemoryBlock({0, total_memory_size});
    }

    RegisteredMemoryBlock DeallocateRegisteredMemoryBlock(
        const RegisteredMemoryBlock& registered_block) {
        IndexedNumber block = registered_block.GetMemoryBlock();
        auto new_block = AdjoinFreeNeighboringMemoryBlocks(block);
        if (new_block == block) {
            is_block_free_[registered_block.id] = true;
            return registered_block;
        } else {
            return RegisterNewFreeMemoryBlock(new_block);
        }
    }

    void AllocateRegisteredMemoryBlock(const RegisteredMemoryBlock& memory_block) {
        is_block_free_[memory_block.id] = false;
    }

    std::pair<RegisteredMemoryBlock, RegisteredMemoryBlock> SplitFreeRegisteredBlock(
        const RegisteredMemoryBlock& memory_block, int32_t left_block_size) {

        AllocateRegisteredMemoryBlock(memory_block);
        return {RegisterNewFreeMemoryBlock({memory_block.begin_position, left_block_size}),
                RegisterNewFreeMemoryBlock({memory_block.begin_position + left_block_size,
                                            memory_block.value - left_block_size})};
    }

    [[nodiscard]] bool IsMemoryBlockFree(const RegisteredMemoryBlock& memory_block) const {
        return is_block_free_[memory_block.id];
    }

    std::optional<RegisteredMemoryBlock> GetFreeRegisteredMemoryBlockByBeginPosition(
        int32_t begin_position) {

        for (auto block_id : block_begin_to_id_table_.GetPreimage(begin_position)) {
            if (memory_blocks_[block_id].begin_position == begin_position and
                is_block_free_[block_id]) {
                return memory_blocks_[block_id];
            }
        }
        return std::nullopt;
    }

    std::optional<RegisteredMemoryBlock> GetFreeRegisteredMemoryBlockByEndPosition(
        int32_t end_position) {

        for (auto block_id : block_end_to_id_table_.GetPreimage(end_position)) {
            if (memory_blocks_[block_id].GetEndPosition() == end_position and
                is_block_free_[block_id]) {
                return memory_blocks_[block_id];
            }
        }
        return std::nullopt;
    }

private:
    HashSet block_begin_to_id_table_;
    HashSet block_end_to_id_table_;
    std::vector<RegisteredMemoryBlock> memory_blocks_;
    std::vector<bool> is_block_free_;

    RegisteredMemoryBlock RegisterNewFreeMemoryBlock(const IndexedNumber& memory_block) {
        RegisteredMemoryBlock registered_memory_block{static_cast<int32_t>(memory_blocks_.size()),
                                                      memory_block};
        memory_blocks_.emplace_back(registered_memory_block);
        is_block_free_.emplace_back(true);

        block_begin_to_id_table_.Insert(registered_memory_block.begin_position,
                                        registered_memory_block.id);

        block_end_to_id_table_.Insert(registered_memory_block.GetEndPosition(),
                                      registered_memory_block.id);

        return registered_memory_block;
    }

    IndexedNumber AdjoinFreeNeighboringMemoryBlocks(IndexedNumber block) {
        auto left_neighbor = GetFreeRegisteredMemoryBlockByEndPosition(block.begin_position);
        if (left_neighbor.has_value()) {
            AllocateRegisteredMemoryBlock(*left_neighbor);
            block.begin_position = left_neighbor->begin_position;
            block.value += left_neighbor->value;
        }

        auto right_neighbor = GetFreeRegisteredMemoryBlockByBeginPosition(block.GetEndPosition());
        if (right_neighbor.has_value()) {
            AllocateRegisteredMemoryBlock(*right_neighbor);
            block.value += right_neighbor->value;
        }

        return block;
    }
};

class MemoryManager {
public:
    Heap<RegisteredMemoryBlock> memory_max_heap;
    MemoryBlockDispenser memory_block_dispenser;
    std::vector<Response> responses;

    MemoryManager(size_t request_count, int32_t total_memory_size)
        : memory_max_heap(request_count, ComparatorMemorySizePositionGreater),
          memory_block_dispenser{total_memory_size} {

        responses.reserve(request_count);
        DeallocateRegisteredMemoryBlock(
            *memory_block_dispenser.GetFreeRegisteredMemoryBlockByBeginPosition(0));
    }

    Response ProcessRequest(Request request) {
        Response response;
        if (auto allocate_request = std::get_if<AllocateRequest>(&request)) {
            response = Allocate(*allocate_request);
        } else {
            response = Deallocate(std::get<DeallocateRequest>(request));
        }

        responses.emplace_back(response);
        return response;
    }

    AllocateResponse Allocate(AllocateRequest request) {
        auto allocated_memory_block = AllocateMemoryFromLargestBlock(request.memory_size);
        if (allocated_memory_block) {
            return ApprovedAllocateResponse{*allocated_memory_block};
        } else {
            return DeclinedAllocateResponse{};
        }
    }

    DeallocateResponse Deallocate(DeallocateRequest deallocate_request) {
        auto allocate_response = std::get<AllocateResponse>(
            responses[deallocate_request.request_id_of_allocate_request]);

        if (auto approved_allocate_response =
                std::get_if<ApprovedAllocateResponse>(&allocate_response)) {

            DeallocateRegisteredMemoryBlock(approved_allocate_response->allocated_memory_block);

            return ApprovedDeallocateResponse{};
        }

        return IgnoredDeallocateResponse{};
    }

private:
    void DeallocateRegisteredMemoryBlock(const RegisteredMemoryBlock& block) {
        auto new_memory_block = memory_block_dispenser.DeallocateRegisteredMemoryBlock(block);
        memory_max_heap.Insert(new_memory_block);
    }

    void PopInvalidMemoryBlocksFromHeap() {
        while (not memory_max_heap.Empty() and
               not memory_block_dispenser.IsMemoryBlockFree(memory_max_heap.Top())) {
            memory_max_heap.Pop();
        }
    }

    std::optional<RegisteredMemoryBlock> AllocateMemoryFromLargestBlock(int32_t memory_size) {
        PopInvalidMemoryBlocksFromHeap();

        if (memory_max_heap.Empty() or memory_max_heap.Top().value < memory_size) {
            return std::nullopt;
        }

        auto block = memory_max_heap.Top();
        memory_max_heap.Pop();

        if (block.value == memory_size) {
            memory_block_dispenser.AllocateRegisteredMemoryBlock(block);
            return block;
        }

        auto [left, right] = memory_block_dispenser.SplitFreeRegisteredBlock(block, memory_size);
        memory_block_dispenser.AllocateRegisteredMemoryBlock(left);
        memory_max_heap.Insert(right);

        return left;
    }
};

OutputType Solve(const InputType& input) {
    MemoryManager memory_manager(input.requests.size(), input.total_memory_size);

    for (auto request : input.requests) {
        memory_manager.ProcessRequest(request);
    }

    return OutputType{memory_manager.responses};
}

namespace test {

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const& message) : message{message.data()} {
    }

    [[nodiscard]] const char* what() const noexcept override {
        return message;
    }

    const char* message;
};

void Check(const std::string& test_case, const std::string& expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    auto expected_output = OutputType{expected};
    if (not(output == expected_output)) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected_output << "\nReceived:\n" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

void PrintSeed() {
    std::cerr << "Seed = " << rng::GetSeed() << '\n';
}

void Test() {
    PrintSeed();
    Check(
        "6 8\n"
        "2\n"
        "3\n"
        "-1\n"
        "3\n"
        "3\n"
        "-5\n"
        "2\n"
        "2\n",
        "0 2 -1 -1 0 -1");
    Check(
        "6 10\n"
        "2\n"
        "3\n"
        "-1\n"
        "3\n"
        "3\n"
        "-5\n"
        "2\n"
        "2\n"
        "1\n"
        "1\n",
        "0 2 -1 -1 0 -1 5 -1");
    Check(
        "2147483647 1\n"
        "2147483647",
        "0");
    Check(
        "2147483646 1\n"
        "2147483647",
        "-1");
    Check(
        "8 11\n"
        "1\n"
        "4\n"
        "-1\n"
        "4\n"
        "2\n"
        "1\n"
        "1\n"
        "1\n"
        "-2\n"
        "5\n"
        "3\n",
        "0 1 -1 5 0 7 -1 -1 1");

    for (int32_t test_case = 0; test_case < 100; ++test_case) {
        int32_t max_memory_size = INT32_MAX;
        int32_t request_count = 10'000;

        InputType input{max_memory_size / (test_case + 1)};
        std::uniform_int_distribution<int32_t> memory_distribution(1, input.total_memory_size);
        std::uniform_int_distribution<int32_t> deallocate_request_id_distribution(1, kPrime);

        std::vector<int32_t> allocation_requests;

        for (int32_t index = 0; index < request_count; ++index) {
            auto dealloc = deallocate_request_id_distribution(rng::GetEngine());
            if (allocation_requests.empty() or dealloc % 4) {

                auto mem = memory_distribution(rng::GetEngine());
                input.requests.emplace_back(AllocateRequest{mem});
                allocation_requests.push_back(index);
            } else {

                dealloc %= static_cast<int32_t>(allocation_requests.size());
                input.requests.emplace_back(DeallocateRequest{allocation_requests[dealloc]});
                allocation_requests.erase(allocation_requests.begin() + dealloc);
            }
        }

        auto my_result = Solve(input);
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
