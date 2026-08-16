#include "next_station/engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace next_station::native;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_map_and_deck() {
    const Map& map = london_map();
    expect(map.stations.size() == kStationCount, "station count mismatch");
    expect(map.edges.size() == kEdgeCount, "edge count mismatch");
    expect(map.district_count == kDistrictCount, "district count mismatch");
    expect(deck().size() == kCardCount, "card count mismatch");
    expect(deck()[10].is_switch, "switch card id mismatch");
}

void check_switch_event() {
    const std::array<std::uint8_t, kColorCount> order = {{0, 1, 2, 3}};
    GameState game(order, 7);
    game.draw_known_cards(std::vector<int>{10, 0});
    expect(game.has_pending(), "known event did not become pending");
    expect(game.pending().count == 2, "switch event card count mismatch");
    expect(game.pending().source_any, "switch event source mode mismatch");
    expect(game.pending().target_symbol == kCircle,
           "switch event target symbol mismatch");
    expect(game.pending().final_card == false, "unexpected final-card flag");
}

void check_chance_probabilities() {
    const std::array<std::uint8_t, kColorCount> order = {{0, 1, 2, 3}};
    GameState game(order, 11);
    std::vector<int> cards;
    cards.push_back(0);
    game.draw_known_cards(cards);
    game.apply_action_unchecked(Action());
    const std::vector<ChanceOutcome> successors = game.public_successors();
    double total = 0.0;
    int switch_outcomes = 0;
    for (std::size_t i = 0; i < successors.size(); ++i) {
        total += successors[i].probability;
        if (successors[i].state.pending().count == 2) ++switch_outcomes;
    }
    expect(std::fabs(total - 1.0) < 1e-12, "chance probabilities do not sum to one");
    expect(switch_outcomes == 9, "switch chance expansion count mismatch");
}

void check_full_game_and_canonical() {
    const std::array<std::uint8_t, kColorCount> order = {{0, 1, 2, 3}};
    GameState game(order, 1234);
    const int initial_total = game.current_total();
    int decisions = 0;
    int reward_sum = 0;
    while (!game.terminated()) {
        expect(!game.has_pending(), "loop starts with a pending event");
        const std::vector<int> remaining_before = [&game]() {
            std::vector<int> values;
            for (int id = 0; id < kCardCount; ++id) {
                if (game.remaining_mask() & (std::uint16_t(1) << id)) values.push_back(id);
            }
            return values;
        }();
        expect(!remaining_before.empty(), "game reached an empty pile");
        const int first = remaining_before.front();
        std::vector<int> event;
        event.push_back(first);
        if (deck()[first].is_switch) {
            expect(remaining_before.size() >= 2, "switch has no second card");
            event.push_back(remaining_before[1]);
        }
        game.draw_known_cards(event);
        const std::vector<Candidate> candidates = game.candidates();
        expect(!candidates.empty(), "candidate list is empty");
        const Candidate& selected = candidates[
            static_cast<std::size_t>(decisions % static_cast<int>(candidates.size()))];
        reward_sum += selected.reward;
        game.apply_action(selected.action);
        ++decisions;
        if (!game.terminated()) {
            const PublicState state = game.canonical();
            const GameState restored = GameState::from_canonical(state);
            expect(restored.canonical_signature() == game.canonical_signature(),
                   "canonical round trip changed public state");
        }
        expect(decisions < 200, "deterministic game did not terminate");
    }
    const FinalScore score = game.final_score();
    expect(score.total >= 0, "final score is negative");
    expect(initial_total + reward_sum == score.total,
           "dense rewards do not telescope to final score");
    expect(game.round_index() == kColorCount - 1, "terminal round index mismatch");
}

void check_advanced_game() {
    GameOptions options;
    options.seed = 9876;
    options.has_order = true;
    options.order = {{0, 1, 2, 3}};
    options.shared_objectives_enabled = true;
    options.objective_count = 2;
    options.objective_cards = {{kEightInterchanges, kSixThamesCrossings}};
    options.pencil_powers_enabled = true;
    options.has_power_assignments = true;
    options.power_assignments = {{
        kDoubleSection, kWildCard, kRailroadSwitch, kCircleStation,
    }};
    GameState game(options);
    const int initial_total = game.current_total();
    int reward_sum = 0;
    int decisions = 0;
    while (!game.terminated()) {
        if (!game.has_pending()) game.draw();
        const std::vector<Action> actions = game.legal_actions();
        Action chosen;
        if (!actions.empty()) {
            chosen = actions.front();
            for (std::size_t index = 0; index < actions.size(); ++index) {
                if (actions[index].power != kNoPower) {
                    chosen = actions[index];
                    break;
                }
            }
            reward_sum += game.score_delta(chosen).total();
        }
        game.apply_action(chosen);
        ++decisions;
        expect(decisions < 250, "advanced game did not terminate");
    }
    const FinalScore score = game.final_score();
    expect(initial_total + reward_sum == score.total,
           "advanced dense rewards do not telescope to final score");
    expect((game.used_power_mask() & (1u << kCircleStation)) != 0,
           "circle power was not resolved at round end");
    expect(score.objective_bonus == score.objectives_completed * 10,
           "advanced objective summary is inconsistent");
}

}  // namespace

int main() {
    try {
        check_map_and_deck();
        check_switch_event();
        check_chance_probabilities();
        check_full_game_and_canonical();
        check_advanced_game();
        std::cout << "next_station_engine_check: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "next_station_engine_check: " << error.what() << "\n";
        return 1;
    }
}
