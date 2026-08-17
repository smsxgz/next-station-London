#ifndef NEXT_STATION_NATIVE_SOLVER_SEARCH_HPP
#define NEXT_STATION_NATIVE_SOLVER_SEARCH_HPP

#include "next_station/engine.hpp"

#include <cstdint>
#include <vector>

namespace next_station {
namespace solver_native {

struct ActionEstimate {
    native::Action action;
    native::ScoreDelta immediate_reward;
    double value;
    std::int64_t visits;
    double standard_error;

    ActionEstimate();
};

struct LookaheadStats {
    std::int64_t decision_nodes;
    std::int64_t chance_nodes;
    std::int64_t chance_outcomes;
    std::int64_t cache_hits;

    LookaheadStats();
};

struct LookaheadResult {
    std::vector<ActionEstimate> estimates;
    LookaheadStats stats;
};

struct MCTSStats {
    std::int64_t simulations;
    std::int64_t decision_nodes;
    std::int64_t tree_chance_samples;
    std::int64_t rollout_chance_samples;
    std::int64_t terminal_rollouts;
    std::int64_t rollout_decisions;
    std::int64_t tree_terminal_hits;
    std::int32_t max_tree_depth;
    double mean_tree_depth;
    double elapsed_seconds;

    MCTSStats();
};

struct MCTSResult {
    std::vector<ActionEstimate> estimates;
    MCTSStats stats;
};

enum class MCTSRolloutPolicy : std::int32_t {
    Greedy = 0,
    Lookahead2 = 1,
    SimpleRandom = 2,
};

LookaheadResult rank_lookahead(
    const native::GameState& root,
    int depth,
    bool specialized_depth_two);

MCTSResult rank_mcts(
    const native::GameState& root,
    int simulations,
    double exploration,
    std::uint64_t seed,
    MCTSRolloutPolicy rollout_policy = MCTSRolloutPolicy::Greedy);

}  // namespace solver_native
}  // namespace next_station

#endif  // NEXT_STATION_NATIVE_SOLVER_SEARCH_HPP
