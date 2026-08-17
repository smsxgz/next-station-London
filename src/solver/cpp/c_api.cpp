#define NEXT_STATION_ENGINE_EXPORTS
#include "c_api.h"
#include "search.hpp"

#include "c_api_internal.hpp"

#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

thread_local std::string g_solver_last_error;

void write_estimate(
    const next_station::solver_native::ActionEstimate& source,
    ns_solver_action_estimate* destination) {
    destination->edge_id = source.action.is_pass() ? -1 : source.action.edge_id;
    destination->source = source.action.is_pass() ? -1 : source.action.source;
    destination->target = source.action.is_pass() ? -1 : source.action.target;
    destination->power = static_cast<int8_t>(source.action.power);
    destination->reward_components[0] = source.immediate_reward.route;
    destination->reward_components[1] = source.immediate_reward.thames;
    destination->reward_components[2] = source.immediate_reward.tourist;
    destination->reward_components[3] = source.immediate_reward.interchange;
    destination->reward_components[4] = source.immediate_reward.objective;
    destination->value = source.value;
    destination->visits = source.visits;
    destination->standard_error = source.standard_error;
}

template <typename Result>
void write_estimates(
    const Result& result,
    ns_solver_action_estimate* estimates,
    int32_t estimate_capacity,
    int32_t* estimate_count) {
    if (estimate_count == 0 || estimate_capacity < 0) {
        throw std::runtime_error("solver estimate output is invalid");
    }
    *estimate_count = static_cast<int32_t>(result.estimates.size());
    if (estimates == 0) return;
    if (estimate_capacity < *estimate_count) {
        throw std::runtime_error("solver estimate buffer is too small");
    }
    for (std::size_t index = 0; index < result.estimates.size(); ++index) {
        write_estimate(result.estimates[index], &estimates[index]);
    }
}

}  // namespace

extern "C" int ns_solver_lookahead(
    ns_game_handle game,
    int32_t depth,
    uint8_t specialized_depth_two,
    ns_solver_action_estimate* estimates,
    int32_t estimate_capacity,
    int32_t* estimate_count,
    ns_solver_lookahead_stats* stats) {
    try {
        g_solver_last_error.clear();
        if (stats == 0) throw std::runtime_error("lookahead stats are required");
        const next_station::solver_native::LookaheadResult result =
            next_station::solver_native::rank_lookahead(
                next_station::native::game_from_c_handle(game),
                depth,
                specialized_depth_two != 0);
        write_estimates(result, estimates, estimate_capacity, estimate_count);
        stats->decision_nodes = result.stats.decision_nodes;
        stats->chance_nodes = result.stats.chance_nodes;
        stats->chance_outcomes = result.stats.chance_outcomes;
        stats->cache_hits = result.stats.cache_hits;
        return 0;
    } catch (const std::exception& error) {
        g_solver_last_error = error.what();
        return 1;
    } catch (...) {
        g_solver_last_error = "unknown native solver error";
        return 1;
    }
}

extern "C" int ns_solver_mcts(
    ns_game_handle game,
    int32_t simulations,
    double exploration,
    uint64_t seed,
    int32_t rollout_policy,
    ns_solver_action_estimate* estimates,
    int32_t estimate_capacity,
    int32_t* estimate_count,
    ns_solver_mcts_stats* stats) {
    try {
        g_solver_last_error.clear();
        if (stats == 0) throw std::runtime_error("MCTS stats are required");
        next_station::solver_native::MCTSRolloutPolicy policy;
        if (rollout_policy == 0) {
            policy = next_station::solver_native::MCTSRolloutPolicy::Greedy;
        } else if (rollout_policy == 1) {
            policy = next_station::solver_native::MCTSRolloutPolicy::Lookahead2;
        } else if (rollout_policy == 2) {
            policy = next_station::solver_native::MCTSRolloutPolicy::SimpleRandom;
        } else {
            throw std::runtime_error("unknown MCTS rollout policy");
        }
        const next_station::solver_native::MCTSResult result =
            next_station::solver_native::rank_mcts(
                next_station::native::game_from_c_handle(game),
                simulations,
                exploration,
                seed,
                policy);
        write_estimates(result, estimates, estimate_capacity, estimate_count);
        stats->simulations = result.stats.simulations;
        stats->decision_nodes = result.stats.decision_nodes;
        stats->tree_chance_samples = result.stats.tree_chance_samples;
        stats->rollout_chance_samples = result.stats.rollout_chance_samples;
        stats->terminal_rollouts = result.stats.terminal_rollouts;
        stats->rollout_decisions = result.stats.rollout_decisions;
        stats->tree_terminal_hits = result.stats.tree_terminal_hits;
        stats->max_tree_depth = result.stats.max_tree_depth;
        stats->mean_tree_depth = result.stats.mean_tree_depth;
        stats->elapsed_seconds = result.stats.elapsed_seconds;
        return 0;
    } catch (const std::exception& error) {
        g_solver_last_error = error.what();
        return 1;
    } catch (...) {
        g_solver_last_error = "unknown native solver error";
        return 1;
    }
}

extern "C" const char* ns_solver_last_error(void) {
    return g_solver_last_error.c_str();
}

extern "C" int32_t ns_solver_action_estimate_size(void) {
    return static_cast<int32_t>(sizeof(ns_solver_action_estimate));
}

extern "C" int32_t ns_solver_lookahead_stats_size(void) {
    return static_cast<int32_t>(sizeof(ns_solver_lookahead_stats));
}

extern "C" int32_t ns_solver_mcts_stats_size(void) {
    return static_cast<int32_t>(sizeof(ns_solver_mcts_stats));
}
