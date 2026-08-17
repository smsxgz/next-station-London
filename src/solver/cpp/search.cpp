#include "search.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace next_station {
namespace solver_native {

namespace {

using native::Action;
using native::GameState;
using native::PencilPower;
using native::PublicDrawList;
using native::ScoreDelta;
using native::Status;

// Keep random rollouts exploratory without making pass the dominant action.
const double kSimpleRandomPassProbability = 0.05;

struct StateKey {
    std::array<std::uint64_t, 19> words;

    bool operator==(const StateKey& other) const {
        return words == other.words;
    }
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const {
        std::size_t result = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
        for (std::size_t index = 0; index < key.words.size(); ++index) {
            const std::size_t value = static_cast<std::size_t>(key.words[index]);
            result ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL)
                + (result << 6) + (result >> 2);
        }
        return result;
    }
};

StateKey state_key(const GameState& game) {
    StateKey result;
    result.words.fill(0);
    const std::array<native::LineState, native::kColorCount>& lines = game.lines();
    std::size_t cursor = 0;
    for (int color = 0; color < native::kColorCount; ++color) {
        result.words[cursor++] = lines[color].station_mask;
        for (int word = 0; word < 3; ++word) {
            result.words[cursor++] = lines[color].edge_mask.words[word];
        }
    }

    std::uint64_t position = game.remaining_mask();
    position |= static_cast<std::uint64_t>(game.round_index()) << 16;
    position |= static_cast<std::uint64_t>(game.underground_count()) << 24;
    position |= static_cast<std::uint64_t>(game.draw_count()) << 32;
    position |= static_cast<std::uint64_t>(game.terminated() ? 1 : 0) << 40;
    for (int index = 0; index < native::kColorCount; ++index) {
        position |= static_cast<std::uint64_t>(game.order()[index])
            << (41 + index * 2);
    }
    result.words[cursor++] = position;

    std::uint64_t advanced = game.shared_objectives_enabled() ? 1u : 0u;
    advanced |= static_cast<std::uint64_t>(game.pencil_powers_enabled() ? 1 : 0)
        << 1;
    advanced |= static_cast<std::uint64_t>(game.shared_objective_mask()) << 8;
    advanced |= static_cast<std::uint64_t>(game.used_power_mask()) << 16;
    advanced |= static_cast<std::uint64_t>(game.completed_objective_mask()) << 24;
    for (int color = 0; color < native::kColorCount; ++color) {
        const int encoded = static_cast<int>(game.power_assignments()[color]) + 1;
        advanced |= static_cast<std::uint64_t>(encoded) << (32 + color * 3);
    }
    advanced |= static_cast<std::uint64_t>(game.double_section_pending() ? 1 : 0)
        << 48;
    advanced |= static_cast<std::uint64_t>(
        static_cast<int>(game.double_target_symbol()) + 1) << 49;
    result.words[cursor++] = advanced;

    std::uint64_t event = game.has_pending() ? 1u : 0u;
    if (game.has_pending()) {
        const native::PendingEvent& pending = game.pending();
        event |= static_cast<std::uint64_t>(pending.count) << 1;
        event |= static_cast<std::uint64_t>(pending.card_ids[0]) << 4;
        event |= static_cast<std::uint64_t>(pending.card_ids[1]) << 8;
        event |= static_cast<std::uint64_t>(
            static_cast<int>(pending.target_symbol) + 1) << 12;
        event |= static_cast<std::uint64_t>(pending.wild ? 1 : 0) << 16;
        event |= static_cast<std::uint64_t>(pending.source_any ? 1 : 0) << 17;
        event |= static_cast<std::uint64_t>(pending.final_card ? 1 : 0) << 18;
    }
    result.words[cursor++] = event;
    return result;
}

struct ValueKey {
    StateKey state;
    int depth;
    bool chance;

    bool operator==(const ValueKey& other) const {
        return depth == other.depth && chance == other.chance
            && state == other.state;
    }
};

struct ValueKeyHash {
    std::size_t operator()(const ValueKey& key) const {
        std::size_t result = StateKeyHash()(key.state);
        result ^= static_cast<std::size_t>(key.depth + 0x9e37)
            + (result << 6) + (result >> 2);
        result ^= static_cast<std::size_t>(key.chance ? 0x85ebca6b : 0xc2b2ae35)
            + (result << 6) + (result >> 2);
        return result;
    }
};

bool action_less(const Action& left, const Action& right) {
    if (left.power != right.power) return left.power < right.power;
    if (left.edge_id != right.edge_id) return left.edge_id < right.edge_id;
    if (left.source != right.source) return left.source < right.source;
    return left.target < right.target;
}

ScoreDelta immediate_reward(const GameState& game, const Action& action) {
    return game.score_delta_for_legal_action(action);
}

std::vector<Action> candidate_actions(const GameState& game) {
    std::vector<Action> result;
    const std::vector<Action> legal = game.legal_actions();
    result.reserve(legal.size() + 1);
    result.push_back(Action());
    result.insert(result.end(), legal.begin(), legal.end());
    return result;
}

class LookaheadSearch {
public:
    LookaheadSearch(int depth, bool specialized_depth_two)
        : depth_(depth), specialized_depth_two_(specialized_depth_two), stats_(),
          values_(), legal_() {}

    LookaheadResult rank(const GameState& root) {
        if (depth_ < 1) throw std::runtime_error("lookahead depth must be positive");
        if (root.status() != Status::Playing || !root.has_pending()) {
            throw std::runtime_error("lookahead requires a pending decision");
        }
        stats_.decision_nodes = 1;
        const std::vector<Action> actions = candidate_actions(root);
        LookaheadResult result;
        result.estimates.reserve(actions.size());
        for (std::size_t index = 0; index < actions.size(); ++index) {
            ActionEstimate estimate;
            estimate.action = actions[index];
            estimate.immediate_reward = immediate_reward(root, actions[index]);
            estimate.value = specialized_depth_two_ && depth_ == 2
                ? depth_two_root_value(root, actions[index], estimate.immediate_reward)
                : action_value(root, actions[index], depth_);
            result.estimates.push_back(estimate);
        }
        result.stats = stats_;
        return result;
    }

private:
    int depth_;
    bool specialized_depth_two_;
    LookaheadStats stats_;
    std::unordered_map<ValueKey, double, ValueKeyHash> values_;
    std::unordered_map<StateKey, std::vector<Action>, StateKeyHash> legal_;

    const std::vector<Action>& cached_actions_for(const GameState& game) {
        const StateKey key = state_key(game);
        const std::unordered_map<StateKey, std::vector<Action>, StateKeyHash>::iterator
            found = legal_.find(key);
        if (found != legal_.end()) return found->second;
        const std::pair<
            std::unordered_map<StateKey, std::vector<Action>, StateKeyHash>::iterator,
            bool> inserted = legal_.emplace(key, candidate_actions(game));
        return inserted.first->second;
    }

    double action_value(
        const GameState& game,
        const Action& action,
        int depth) {
        const double immediate = static_cast<double>(
            immediate_reward(game, action).total());
        GameState child = game.copy_public();
        child.apply_action_unchecked(action);
        if (child.terminated()) return immediate;
        if (child.has_pending()) {
            return immediate + decision_value(child, depth);
        }
        if (depth == 1) return immediate;
        return immediate + chance_value(child, depth - 1);
    }

    double decision_value(const GameState& game, int depth) {
        if (depth < 1 || game.terminated()) return 0.0;
        if (!game.has_pending()) {
            throw std::runtime_error("decision expansion requires a pending event");
        }
        const ValueKey key = {state_key(game), depth, false};
        const std::unordered_map<ValueKey, double, ValueKeyHash>::const_iterator found =
            values_.find(key);
        if (found != values_.end()) {
            ++stats_.cache_hits;
            return found->second;
        }

        ++stats_.decision_nodes;
        const std::vector<Action> actions = candidate_actions(game);
        double value = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < actions.size(); ++index) {
            value = std::max(value, action_value(game, actions[index], depth));
        }
        values_.emplace(key, value);
        return value;
    }

    double chance_value(const GameState& game, int depth) {
        if (depth < 1 || game.terminated()) return 0.0;
        if (game.has_pending()) {
            throw std::runtime_error("chance expansion requires a resolved event");
        }
        const ValueKey key = {state_key(game), depth, true};
        const std::unordered_map<ValueKey, double, ValueKeyHash>::const_iterator found =
            values_.find(key);
        if (found != values_.end()) {
            ++stats_.cache_hits;
            return found->second;
        }

        ++stats_.chance_nodes;
        double expected = 0.0;
        double probability_sum = 0.0;
        const PublicDrawList draws = game.public_draws();
        for (std::size_t index = 0; index < draws.count; ++index) {
            const native::PublicDraw& draw = draws.items[index];
            const GameState outcome = game.public_successor(draw);
            ++stats_.chance_outcomes;
            probability_sum += draw.probability;
            expected += draw.probability * decision_value(outcome, depth);
        }
        if (std::fabs(probability_sum - 1.0) > 1e-12) {
            throw std::runtime_error("chance probabilities do not sum to one");
        }
        values_.emplace(key, expected);
        return expected;
    }

    bool double_section_available(const GameState& game) const {
        if (!game.pencil_powers_enabled()) return false;
        const int color = game.order()[game.round_index()];
        return game.power_assignments()[color] == native::kDoubleSection
            && (game.used_power_mask() & (1u << native::kDoubleSection)) == 0;
    }

    double final_card_action_value(
        const GameState& game,
        const Action& action) {
        const double immediate = static_cast<double>(
            immediate_reward(game, action).total());
        if (action.is_pass() || !double_section_available(game)) return immediate;

        GameState child = game.copy_public();
        child.apply_action_unchecked(action);
        if (child.terminated() || !child.has_pending()) return immediate;
        return immediate + decision_value(child, 1);
    }

    double depth_two_root_value(
        const GameState& game,
        const Action& action,
        const ScoreDelta& reward) {
        const double immediate = static_cast<double>(reward.total());
        GameState child = game.copy_public();
        child.apply_action_unchecked(action);
        if (child.terminated()) return immediate;
        if (child.has_pending()) {
            return immediate + decision_value(child, 2);
        }

        ++stats_.chance_nodes;
        double expected = 0.0;
        double probability_sum = 0.0;
        const PublicDrawList draws = child.public_draws();
        for (std::size_t index = 0; index < draws.count; ++index) {
            const native::PublicDraw& draw = draws.items[index];
            const GameState outcome = child.public_successor(draw);
            ++stats_.chance_outcomes;
            ++stats_.decision_nodes;
            probability_sum += draw.probability;
            const std::vector<Action>& actions = cached_actions_for(outcome);
            double best = -std::numeric_limits<double>::infinity();
            for (std::size_t action_index = 0;
                 action_index < actions.size(); ++action_index) {
                best = std::max(
                    best,
                    final_card_action_value(outcome, actions[action_index]));
            }
            expected += draw.probability * best;
        }
        if (std::fabs(probability_sum - 1.0) > 1e-12) {
            throw std::runtime_error("chance probabilities do not sum to one");
        }
        return immediate + expected;
    }
};

struct MCTSActionStats {
    std::int64_t visits;
    double value_sum;
    double value_square_sum;

    MCTSActionStats() : visits(0), value_sum(0.0), value_square_sum(0.0) {}

    double mean() const {
        return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
    }

    double standard_error() const {
        if (visits < 2) return 0.0;
        const double count = static_cast<double>(visits);
        const double variance = (
            value_square_sum - value_sum * value_sum / count)
            / static_cast<double>(visits - 1);
        return std::sqrt(std::max(0.0, variance) / count);
    }

    void update(double value) {
        ++visits;
        value_sum += value;
        value_square_sum += value * value;
    }
};

struct DecisionNode {
    std::vector<Action> actions;
    std::vector<MCTSActionStats> action_stats;
    std::int64_t visits;

    explicit DecisionNode(const GameState& game)
        : actions(candidate_actions(game)), action_stats(actions.size()), visits(0) {}
};

std::size_t random_index(std::mt19937_64* rng, std::size_t count) {
    if (count == 0) throw std::runtime_error("cannot sample an empty collection");
    std::uniform_int_distribution<std::size_t> distribution(0, count - 1);
    return distribution(*rng);
}

std::size_t select_action_index(
    const DecisionNode& node,
    double exploration,
    std::mt19937_64* rng) {
    std::array<std::size_t, native::kEdgeCount + 1> candidates;
    std::size_t candidate_count = 0;
    for (std::size_t index = 0; index < node.action_stats.size(); ++index) {
        if (node.action_stats[index].visits == 0) {
            candidates[candidate_count++] = index;
        }
    }
    if (candidate_count != 0) {
        return candidates[random_index(rng, candidate_count)];
    }

    const double log_visits = std::log(static_cast<double>(node.visits));
    double best = -std::numeric_limits<double>::infinity();
    candidate_count = 0;
    for (std::size_t index = 0; index < node.action_stats.size(); ++index) {
        const MCTSActionStats& stats = node.action_stats[index];
        const double value = stats.mean() + exploration * std::sqrt(
            log_visits / static_cast<double>(stats.visits));
        if (value > best + 1e-12) {
            best = value;
            candidate_count = 0;
            candidates[candidate_count++] = index;
        } else if (std::fabs(value - best) <= 1e-12) {
            candidates[candidate_count++] = index;
        }
    }
    return candidates[random_index(rng, candidate_count)];
}

struct EventScratch {
    std::array<int, native::kCardCount> remaining;
    std::vector<Action> legal_actions;

    EventScratch() : remaining(), legal_actions() {
        legal_actions.reserve(native::kEdgeCount);
    }
};

void sample_public_event(
    GameState* game,
    std::mt19937_64* rng,
    EventScratch* scratch) {
    std::size_t remaining_count = 0;
    const std::uint16_t mask = game->remaining_mask();
    for (int card = 0; card < native::kCardCount; ++card) {
        if (mask & (std::uint16_t(1) << card)) {
            scratch->remaining[remaining_count++] = card;
        }
    }
    if (remaining_count == 0) {
        throw std::runtime_error("cannot sample an empty card pile");
    }
    const std::size_t first_index = random_index(rng, remaining_count);
    const int first = scratch->remaining[first_index];
    if (native::deck()[first].is_switch) {
        if (remaining_count == 1) {
            throw std::runtime_error("switch card has no following card");
        }
        std::size_t second_index = random_index(rng, remaining_count - 1);
        if (second_index >= first_index) ++second_index;
        game->draw_known_cards(first, scratch->remaining[second_index]);
    } else {
        game->draw_known_cards(first);
    }
}

Action deterministic_greedy_action(
    const GameState& game,
    const std::vector<Action>* first_actions,
    std::size_t first_action_index,
    std::vector<Action>* legal_scratch) {
    if (first_actions == 0) game.legal_actions(legal_scratch);
    const std::vector<Action>& legal = first_actions == 0
        ? *legal_scratch : *first_actions;
    Action best_action;
    bool has_best = false;
    int best_value = 0;
    for (std::size_t index = first_action_index; index < legal.size(); ++index) {
        const int value = immediate_reward(game, legal[index]).total();
        if (value > best_value
            || (value == best_value
                && (!has_best || action_less(legal[index], best_action)))) {
            best_action = legal[index];
            best_value = value;
            has_best = true;
        }
    }
    return has_best ? best_action : Action();
}

struct RolloutResult {
    int total;
    std::int64_t decisions;
    std::int64_t chance_samples;
};

RolloutResult complete_greedy_rollout(
    GameState* game,
    std::mt19937_64* rng,
    EventScratch* event_scratch,
    const std::vector<Action>& first_actions) {
    RolloutResult result = {0, 0, 0};
    const std::vector<Action>* legal = &first_actions;
    std::size_t first_action_index = 1;
    while (!game->terminated()) {
        if (!game->has_pending()) {
            sample_public_event(game, rng, event_scratch);
            ++result.chance_samples;
        }
        const Action action = deterministic_greedy_action(
            *game, legal, first_action_index, &event_scratch->legal_actions);
        legal = 0;
        first_action_index = 0;
        game->apply_action_unchecked(action);
        ++result.decisions;
    }
    result.total = game->final_score().total;
    return result;
}

Action deterministic_lookahead2_action(
    const GameState& game,
    LookaheadSearch* search) {
    const LookaheadResult ranked = search->rank(game);
    Action best_action;
    bool has_best = false;
    double best_value = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < ranked.estimates.size(); ++index) {
        const ActionEstimate& estimate = ranked.estimates[index];
        if (estimate.value > best_value + 1e-12
            || (std::fabs(estimate.value - best_value) <= 1e-12
                && (!has_best || action_less(estimate.action, best_action)))) {
            best_action = estimate.action;
            best_value = estimate.value;
            has_best = true;
        }
    }
    if (!has_best) throw std::runtime_error("lookahead rollout found no action");
    return best_action;
}

RolloutResult complete_lookahead2_rollout(
    GameState* game,
    std::mt19937_64* rng,
    EventScratch* event_scratch) {
    RolloutResult result = {0, 0, 0};
    LookaheadSearch search(2, true);
    while (!game->terminated()) {
        if (!game->has_pending()) {
            sample_public_event(game, rng, event_scratch);
            ++result.chance_samples;
        }
        const Action action = deterministic_lookahead2_action(*game, &search);
        game->apply_action_unchecked(action);
        ++result.decisions;
    }
    result.total = game->final_score().total;
    return result;
}

RolloutResult complete_simple_random_rollout(
    GameState* game,
    std::mt19937_64* rng,
    EventScratch* event_scratch,
    const std::vector<Action>& first_actions) {
    RolloutResult result = {0, 0, 0};
    const std::vector<Action>* first = &first_actions;
    std::bernoulli_distribution pass_distribution(
        kSimpleRandomPassProbability);
    while (!game->terminated()) {
        if (!game->has_pending()) {
            sample_public_event(game, rng, event_scratch);
            ++result.chance_samples;
        }
        const std::vector<Action>* actions = first;
        const std::size_t first_action_index = first == &first_actions ? 1 : 0;
        if (actions == 0) {
            event_scratch->legal_actions.clear();
            game->legal_actions(&event_scratch->legal_actions);
            actions = &event_scratch->legal_actions;
        }
        Action action;
        if (actions->size() > first_action_index) {
            if (!pass_distribution(*rng)) {
                const std::size_t offset = random_index(
                    rng,
                    actions->size() - first_action_index);
                action = (*actions)[first_action_index + offset];
            }
        }
        game->apply_action_unchecked(action);
        ++result.decisions;
        first = 0;
    }
    result.total = game->final_score().total;
    return result;
}

}  // namespace

ActionEstimate::ActionEstimate()
    : action(), immediate_reward(), value(0.0), visits(0), standard_error(0.0) {}

LookaheadStats::LookaheadStats()
    : decision_nodes(0), chance_nodes(0), chance_outcomes(0), cache_hits(0) {}

MCTSStats::MCTSStats()
    : simulations(0), decision_nodes(0), tree_chance_samples(0),
      rollout_chance_samples(0), terminal_rollouts(0), rollout_decisions(0),
      tree_terminal_hits(0), max_tree_depth(0), mean_tree_depth(0.0),
      elapsed_seconds(0.0) {}

LookaheadResult rank_lookahead(
    const GameState& root,
    int depth,
    bool specialized_depth_two) {
    return LookaheadSearch(depth, specialized_depth_two).rank(root);
}

MCTSResult rank_mcts(
    const GameState& root,
    int simulations,
    double exploration,
    std::uint64_t seed,
    MCTSRolloutPolicy rollout_policy) {
    if (root.status() != Status::Playing || !root.has_pending()) {
        throw std::runtime_error("MCTS requires a pending decision");
    }
    if (simulations < 1) throw std::runtime_error("MCTS simulations must be positive");
    if (!std::isfinite(exploration) || exploration < 0.0) {
        throw std::runtime_error("MCTS exploration must be finite and non-negative");
    }
    if (rollout_policy != MCTSRolloutPolicy::Greedy
        && rollout_policy != MCTSRolloutPolicy::Lookahead2
        && rollout_policy != MCTSRolloutPolicy::SimpleRandom) {
        throw std::runtime_error("unknown MCTS rollout policy");
    }

    const std::chrono::steady_clock::time_point started =
        std::chrono::steady_clock::now();
    std::mt19937_64 rng(seed);
    const int baseline = root.current_total();
    std::deque<DecisionNode> nodes;
    nodes.emplace_back(root);
    DecisionNode* root_node = &nodes.back();
    std::unordered_map<StateKey, DecisionNode*, StateKeyHash> table;
    table.reserve(static_cast<std::size_t>(simulations) + 1);
    table.emplace(state_key(root), root_node);

    MCTSStats stats;
    stats.simulations = simulations;
    std::int64_t tree_depth_sum = 0;
    std::vector<std::pair<DecisionNode*, std::size_t> > path;
    path.reserve(64);
    EventScratch event_scratch;
    for (int simulation_index = 0;
         simulation_index < simulations; ++simulation_index) {
        GameState simulation = root.copy_public();
        DecisionNode* node = root_node;
        path.clear();
        int tree_depth = 0;
        int terminal_score = 0;

        while (true) {
            const std::size_t action_index = select_action_index(
                *node, exploration, &rng);
            path.push_back(std::make_pair(node, action_index));
            ++tree_depth;
            simulation.apply_action_unchecked(node->actions[action_index]);
            if (simulation.terminated()) {
                terminal_score = simulation.final_score().total;
                ++stats.tree_terminal_hits;
                break;
            }

            if (!simulation.has_pending()) {
                sample_public_event(&simulation, &rng, &event_scratch);
                ++stats.tree_chance_samples;
            }
            const StateKey child_key = state_key(simulation);
            const std::unordered_map<
                StateKey,
                DecisionNode*,
                StateKeyHash>::iterator found = table.find(child_key);
            if (found == table.end()) {
                nodes.emplace_back(simulation);
                DecisionNode* child = &nodes.back();
                table.emplace(child_key, child);
                RolloutResult rollout;
                if (rollout_policy == MCTSRolloutPolicy::Greedy) {
                    rollout = complete_greedy_rollout(
                        &simulation, &rng, &event_scratch, child->actions);
                } else if (rollout_policy == MCTSRolloutPolicy::Lookahead2) {
                    rollout = complete_lookahead2_rollout(
                        &simulation, &rng, &event_scratch);
                } else if (rollout_policy == MCTSRolloutPolicy::SimpleRandom) {
                    rollout = complete_simple_random_rollout(
                        &simulation, &rng, &event_scratch, child->actions);
                } else {
                    throw std::runtime_error("unknown MCTS rollout policy");
                }
                terminal_score = rollout.total;
                ++stats.terminal_rollouts;
                stats.rollout_decisions += rollout.decisions;
                stats.rollout_chance_samples += rollout.chance_samples;
                break;
            }
            node = found->second;
        }

        const double gain = static_cast<double>(terminal_score - baseline);
        for (std::size_t index = 0; index < path.size(); ++index) {
            ++path[index].first->visits;
            path[index].first->action_stats[path[index].second].update(gain);
        }
        tree_depth_sum += tree_depth;
        stats.max_tree_depth = std::max(stats.max_tree_depth, tree_depth);
    }

    if (root_node->visits != simulations) {
        throw std::runtime_error("MCTS root visit count does not match its budget");
    }
    stats.decision_nodes = static_cast<std::int64_t>(table.size());
    stats.mean_tree_depth = static_cast<double>(tree_depth_sum)
        / static_cast<double>(simulations);
    stats.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    MCTSResult result;
    result.stats = stats;
    result.estimates.reserve(root_node->actions.size());
    for (std::size_t index = 0; index < root_node->actions.size(); ++index) {
        ActionEstimate estimate;
        estimate.action = root_node->actions[index];
        estimate.value = root_node->action_stats[index].mean();
        estimate.visits = root_node->action_stats[index].visits;
        estimate.standard_error = root_node->action_stats[index].standard_error();
        result.estimates.push_back(estimate);
    }
    return result;
}

}  // namespace solver_native
}  // namespace next_station
