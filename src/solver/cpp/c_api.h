#ifndef NEXT_STATION_NATIVE_SOLVER_C_API_H
#define NEXT_STATION_NATIVE_SOLVER_C_API_H

#include "next_station/c_api.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)

typedef struct ns_solver_action_estimate {
    int32_t edge_id;
    int32_t source;
    int32_t target;
    int8_t power;
    int32_t reward_components[5];
    double value;
    int64_t visits;
    double standard_error;
} ns_solver_action_estimate;

typedef struct ns_solver_lookahead_stats {
    int64_t decision_nodes;
    int64_t chance_nodes;
    int64_t chance_outcomes;
    int64_t cache_hits;
} ns_solver_lookahead_stats;

typedef struct ns_solver_mcts_stats {
    int64_t simulations;
    int64_t decision_nodes;
    int64_t tree_chance_samples;
    int64_t rollout_chance_samples;
    int64_t terminal_rollouts;
    int64_t rollout_decisions;
    int64_t tree_terminal_hits;
    int32_t max_tree_depth;
    double mean_tree_depth;
    double elapsed_seconds;
} ns_solver_mcts_stats;

#pragma pack(pop)

NS_ENGINE_API int ns_solver_lookahead(
    ns_game_handle game,
    int32_t depth,
    uint8_t specialized_depth_two,
    ns_solver_action_estimate* estimates,
    int32_t estimate_capacity,
    int32_t* estimate_count,
    ns_solver_lookahead_stats* stats);

NS_ENGINE_API int ns_solver_mcts(
    ns_game_handle game,
    int32_t simulations,
    double exploration,
    uint64_t seed,
    int32_t rollout_policy,
    ns_solver_action_estimate* estimates,
    int32_t estimate_capacity,
    int32_t* estimate_count,
    ns_solver_mcts_stats* stats);

NS_ENGINE_API const char* ns_solver_last_error(void);
NS_ENGINE_API int32_t ns_solver_action_estimate_size(void);
NS_ENGINE_API int32_t ns_solver_lookahead_stats_size(void);
NS_ENGINE_API int32_t ns_solver_mcts_stats_size(void);

#ifdef __cplusplus
}
#endif

#endif  // NEXT_STATION_NATIVE_SOLVER_C_API_H
