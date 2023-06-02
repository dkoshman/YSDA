#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

namespace rng {
std::random_device &GetDevice() {
    static std::random_device random_device;
    return random_device;
}

std::mt19937 &GetGenerator() {
    auto &device = GetDevice();
    static std::mt19937 generator(device());
    return generator;
}
}  // namespace rng

class Light {
public:
    [[nodiscard]] bool IsOn() const {
        return is_on;
    }

    [[nodiscard]] bool IsOff() const {
        return not IsOn();
    }

    void TurnOn() {
        is_on = true;
    }

    void TurnOff() {
        is_on = false;
    }

    bool is_on = false;
};

struct PrisonerInput {
    int32_t day_number = 0;
    Light *light = nullptr;
};

enum class PrisonerClaim { claim_nothing, claim_that_everyone_has_been_in_the_room };

class PrisonerBase {
public:
    PrisonerBase(int32_t prisoner_id, int32_t n_prisoners)
        : prisoner_id{prisoner_id}, n_prisoners{n_prisoners} {
    }

    virtual PrisonerClaim TakeAction(PrisonerInput input) = 0;

    int32_t prisoner_id = 0;
    int32_t n_prisoners = 0;
};

class FalsePrisonerClaimException : public std::exception {};

template <class Prisoner>
class Prison {
public:
    explicit Prison(int32_t n_prisoners)
        : n_prisoners{n_prisoners},
          distribution_(0, n_prisoners - 1),
          prisoners_have_been_in_the_room_indicators(n_prisoners) {
        for (int32_t i = 0; i < n_prisoners; ++i) {
            prisoners.emplace_back(i, n_prisoners);
        }
    }

    bool HaveAllPrisonersBeenInTheRoom() {
        return std::all_of(prisoners_have_been_in_the_room_indicators.begin(),
                           prisoners_have_been_in_the_room_indicators.end(),
                           [](bool x) { return x; });
    }

    PrisonerClaim NextDay() {
        auto prisoner_id = distribution_(rng::GetGenerator());
        prisoners_have_been_in_the_room_indicators[prisoner_id] = true;
        auto prisoner_claim = prisoners[prisoner_id].TakeAction({day_number, &light});
        ++day_number;
        return prisoner_claim;
    }

    int32_t Run() {
        while (true) {
            auto prisoner_claim = NextDay();
            if (prisoner_claim == PrisonerClaim::claim_that_everyone_has_been_in_the_room) {
                if (HaveAllPrisonersBeenInTheRoom()) {
                    return day_number;
                } else {
                    throw FalsePrisonerClaimException{};
                }
            }
        }
    }

    [[maybe_unused]] int32_t n_prisoners = 0;
    int32_t day_number = 0;
    Light light = Light{};
    std::vector<Prisoner> prisoners;
    std::vector<bool> prisoners_have_been_in_the_room_indicators;

private:
    std::uniform_int_distribution<int32_t> distribution_;
};

class DedicatedCounterPrisoner : public PrisonerBase {
public:
    using PrisonerBase::PrisonerBase;

    PrisonerClaim TakeAction(PrisonerInput input) override {
        if (prisoner_id == 0) {
            if (input.light->IsOn()) {
                input.light->TurnOff();
                ++times_turned_off_the_light;
            }
            if (times_turned_off_the_light == n_prisoners - 1) {
                return PrisonerClaim::claim_that_everyone_has_been_in_the_room;
            }
        } else {
            if (not has_turned_on_the_light and input.light->IsOff()) {
                input.light->TurnOn();
                has_turned_on_the_light = true;
            }
        }
        return PrisonerClaim::claim_nothing;
    }

    bool has_turned_on_the_light = false;
    int32_t times_turned_off_the_light = 0;
};

class TokenPrisoner : public PrisonerBase {
public:
    TokenPrisoner(int32_t prisoner_id, int32_t n_prisoners, double stage_probability = 0.95,
                  double after_first_cycle_stage_length_multiplier = 0.5)
        : PrisonerBase{prisoner_id, n_prisoners} {

        n_stages = GetClosestNotSmallerPowerOf2(n_prisoners);
        auto n_prisoners_with_2_tokens_initially = (1 << n_stages) - n_prisoners;
        auto n_prisoners_with_1_token_initially = n_prisoners - n_prisoners_with_2_tokens_initially;
        n_tokens = prisoner_id < n_prisoners_with_2_tokens_initially ? 2 : 1;

        for (int i = 1; i <= n_stages; i++) {
            if (i == 1) {
                auto stage_1_length =
                    ComputeNumberOfDaysSoThatKPrisonersVisitTheRoomWithGivenProbability(
                        n_prisoners_with_1_token_initially, stage_probability);
                first_cycle_stage_lengths.push_back(stage_1_length);

            } else {
                auto light_in_tokens_value_at_stage_i = 1 << (i - 1);
                int32_t expected_number_of_prisoners_with_tokens =
                    (1 << n_stages) / light_in_tokens_value_at_stage_i;
                auto stage_length =
                    ComputeNumberOfDaysSoThatKPrisonersVisitTheRoomWithGivenProbability(
                        expected_number_of_prisoners_with_tokens, stage_probability);
                first_cycle_stage_lengths.push_back(stage_length);
            }
        }

        for (auto i : first_cycle_stage_lengths) {
            after_first_cycle_stage_lengths.push_back(
                static_cast<int32_t>(i * after_first_cycle_stage_length_multiplier));
        }
    }

    static int32_t GetClosestNotSmallerPowerOf2(int32_t number) {
        return std::ceil(log2(number));
    }

    template <class T>
    static T NChooseK(T n, T k) {
        if (n < k) {
            return 0;
        }
        k = std::min(k, n - k);
        T result = 1;
        for (int i = 1; i <= k; i++) {
            result = result * (n - k + i) / i;
        }
        return result;
    }

    static double ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
        int32_t k_prisoners, int32_t n_days, int32_t n_prisoners) {
        if (n_days < k_prisoners or n_prisoners < k_prisoners) {
            return 0;
        }

        double result = 0;
        for (int32_t i = 0; i <= k_prisoners; i++) {
            result += NChooseK<double>(k_prisoners, i) * std::pow(-1, i) *
                      std::exp(n_days * std::log1p(static_cast<double>(-i) / n_prisoners));
        }

        if (result < -1.0e-3 or result > 1) {
            throw std::runtime_error("Unstable probability calculation.");
        }

        return std::max(0.0, result);
    }

    int32_t ComputeNumberOfDaysSoThatKPrisonersVisitTheRoomWithGivenProbability(
        int32_t k_prisoners, double target_probability) {
        if (k_prisoners > n_prisoners) {
            throw std::invalid_argument{
                "Requested number of prisoners is greater than total number."};
        }
        int32_t galloping_bin_search_power = 0;
        while (ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
                   k_prisoners, 1 << (galloping_bin_search_power + 1), n_prisoners) <
               target_probability) {
            ++galloping_bin_search_power;
        }

        auto low = 1 << galloping_bin_search_power;
        auto high = 1 << (galloping_bin_search_power + 1);
        while (low < high) {
            int32_t mid = (low + high) / 2;
            auto probability = ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
                k_prisoners, mid, n_prisoners);
            if (probability < target_probability) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        assert(ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
                   k_prisoners, low, n_prisoners) >= target_probability);
        return low;
    }

    [[nodiscard]] int32_t GetStageIndex(int32_t day_number) const {
        if (n_prisoners == 1) {
            return 0;
        }

        int32_t accumulated_days = 0;
        for (int i = 0; i < n_stages; ++i) {
            accumulated_days += first_cycle_stage_lengths[i];
            if (day_number < accumulated_days) {
                return i;
            }
        }

        auto stage_index = 0;
        while (true) {
            accumulated_days += after_first_cycle_stage_lengths[stage_index];
            if (day_number < accumulated_days) {
                return stage_index;
            }
            stage_index = (stage_index + 1) % n_stages;
        }
        assert(false);
    }

    [[nodiscard]] bool IsLastDayOfTheStage(int32_t day_number) const {
        return GetStageIndex(day_number) != GetStageIndex(day_number + 1);
    }

    void MaybeTurnOffLight(PrisonerInput input) {
        if (input.light->IsOff()) {
            return;
        }
        auto stage_index = GetStageIndex(input.day_number);
        auto light_in_tokens_value = 1 << stage_index;
        bool have_matching_bit = n_tokens & light_in_tokens_value;
        if (IsLastDayOfTheStage(input.day_number) or have_matching_bit) {
            n_tokens += light_in_tokens_value;
            input.light->TurnOff();
        }
    }

    void MaybeTurnOnLight(PrisonerInput input) {
        if (input.light->IsOn()) {
            return;
        }
        auto next_day_stage_index = GetStageIndex(input.day_number + 1);
        auto next_day_light_in_tokens_value = 1 << next_day_stage_index;
        auto have_matching_bit = n_tokens & next_day_light_in_tokens_value;
        if (have_matching_bit) {
            n_tokens -= next_day_light_in_tokens_value;
            input.light->TurnOn();
        }
    }

    [[nodiscard]] bool ShouldClaimThatEveryoneHasBeenInTheRoom() const {
        return n_tokens == 1 << GetClosestNotSmallerPowerOf2(n_prisoners);
    }

    PrisonerClaim TakeAction(PrisonerInput input) override {
        if (n_prisoners == 1) {
            return PrisonerClaim::claim_that_everyone_has_been_in_the_room;
        }

        MaybeTurnOffLight(input);
        MaybeTurnOnLight(input);

        if (ShouldClaimThatEveryoneHasBeenInTheRoom()) {
            return PrisonerClaim::claim_that_everyone_has_been_in_the_room;
        } else {
            return PrisonerClaim::claim_nothing;
        }
    }

    int32_t n_tokens = 0;
    int32_t n_stages = 0;
    std::vector<int32_t> first_cycle_stage_lengths;
    std::vector<int32_t> after_first_cycle_stage_lengths;
};

namespace test {

int64_t Factorial(int32_t n) {
    int64_t result = 1;
    for (int32_t i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template <class T>
bool IsClose(T first, T second, T eps = 1.0e-7) {
    return std::abs(first - second) < eps;
}

template <class Prisoner>
void Test() {

    for (int32_t n = 1; n <= 10; ++n) {
        for (int32_t k = 1; k <= n; ++k) {
            auto expected_n_choose_k = Factorial(n) / Factorial(k) / Factorial(n - k);
            auto n_choose_k = TokenPrisoner::NChooseK(n, k);
            assert(n_choose_k == expected_n_choose_k);
        }
    }

    auto n_choose_k = TokenPrisoner::NChooseK<double>(64, 32);
    assert(IsClose(n_choose_k, 1832624140942590534.0, 1.0e+3));

    {
        auto n_prisoners = 5;
        auto k_prisoners = 2;
        auto n_days = 2;
        double expected_probability = 0.08;
        double probability =
            TokenPrisoner::ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
                k_prisoners, n_days, n_prisoners);
        assert(IsClose(probability, expected_probability));

        n_prisoners = 4;
        k_prisoners = 2;
        n_days = 3;
        expected_probability = 9.0 / 32;
        probability = TokenPrisoner::ComputeProbabilityThatKFixedPrisonersWereInTheRoomDuringNDays(
            k_prisoners, n_days, n_prisoners);
        assert(IsClose(probability, expected_probability));
    }

    for (int32_t n_prisoners = 1; n_prisoners <= 100; n_prisoners *= 2) {
        Prison<Prisoner>(n_prisoners).Run();
    }
    Prison<Prisoner>(100).Run();
}
}  // namespace test

template <class Prisoner>
void RunPrisonSimulations(int32_t n_prisoners, int32_t n_simulations) {

    test::Test<Prisoner>();

    std::vector<double> days_prison_ran_for;
    for (int i = 0; i < n_simulations; ++i) {
        auto prison = Prison<Prisoner>(n_prisoners);
        days_prison_ran_for.push_back(prison.Run());
    }

    double days_mean =
        std::reduce(days_prison_ran_for.begin(), days_prison_ran_for.end()) / n_simulations;
    double days_std = 0;
    for (auto i : days_prison_ran_for) {
        days_std += (i - days_mean) * (i - days_mean);
    }
    days_std = std::sqrt(days_std / n_simulations);

    std::cout << "Days mean:\t" << static_cast<int32_t>(days_mean);
    std::cout << "\nDays std:\t" << days_std;
}

int main(int argc, char *argv[]) {
    // Usage: [prisoner_class_name] [n_prisoners] [n_simulations]

    std::string prisoner_class_name = "DedicatedCounterPrisoner";
    int32_t n_prisoners = 100;
    int32_t n_simulations = 1000;

    if (argc >= 2) {
        prisoner_class_name = argv[1];
    }
    if (argc >= 3) {
        std::istringstream iss{argv[2]};
        iss >> n_prisoners;
    }
    if (argc >= 4) {
        std::istringstream iss{argv[3]};
        iss >> n_simulations;
    }

    if (prisoner_class_name == "DedicatedCounterPrisoner") {
        RunPrisonSimulations<DedicatedCounterPrisoner>(n_prisoners, n_simulations);
    } else if (prisoner_class_name == "TokenPrisoner") {
        RunPrisonSimulations<TokenPrisoner>(n_prisoners, n_simulations);
    } else {
        throw std::invalid_argument{"Unknown Prisoner class name."};
    }

    return 0;
}
