#ifndef NEXT_STATION_NATIVE_C_API_H
#define NEXT_STATION_NATIVE_C_API_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(NEXT_STATION_ENGINE_EXPORTS)
#    define NS_ENGINE_API __declspec(dllexport)
#  else
#    define NS_ENGINE_API __declspec(dllimport)
#  endif
#else
#  define NS_ENGINE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)

typedef struct ns_public_state {
    uint64_t line_station_masks[4];
    uint64_t line_edge_words[4][3];
    uint16_t remaining_mask;
    uint8_t order[4];
    uint8_t round_index;
    uint8_t underground_count;
    uint8_t draw_count;
    uint8_t terminated;
} ns_public_state;

typedef struct ns_outcome {
    double probability;
    int32_t candidate_offset;
    int32_t candidate_count;
    uint8_t card_ids[2];
    uint8_t card_count;
} ns_outcome;

typedef struct ns_candidate {
    int32_t action_index;
    int32_t source;
    int32_t target;
    int32_t reward;
    ns_public_state afterstate;
} ns_candidate;

typedef struct ns_state_metrics {
    uint16_t line_district_masks[4];
    uint8_t line_station_counts[4][13];
    int32_t line_district_counts[4];
    int32_t line_max_stations[4];
    int32_t line_routes[4];
    int32_t line_thames_crossings[4];
    int32_t line_tourist_visits[4];
    int32_t partial_components[6];
    uint8_t interchange_counts[5];
    int32_t current_total;
    int32_t final_total;
} ns_state_metrics;

typedef struct ns_game_snapshot {
    ns_public_state state;
    uint64_t line_leaf_masks[4];
    uint8_t shared_objectives_enabled;
    uint8_t pencil_powers_enabled;
    uint8_t objective_cards[2];
    uint8_t shared_objective_mask;
    int8_t power_assignments[4];
    uint8_t used_power_mask;
    uint8_t completed_objective_mask;
    uint8_t double_section_pending;
    int8_t double_target_symbol;
    uint8_t has_pending;
    uint8_t pending_card_ids[2];
    uint8_t pending_card_count;
    int8_t pending_target_symbol;
    uint8_t pending_wild;
    uint8_t pending_source_any;
    uint8_t pending_final_card;
    int32_t partial_components[6];
    int32_t score_summary[10];
    uint8_t round_score_count;
    int32_t round_scores[4][6];
} ns_game_snapshot;

typedef struct ns_game_action {
    int32_t edge_id;
    int32_t source;
    int32_t target;
    int8_t power;
    int32_t reward_components[5];
} ns_game_action;

typedef struct ns_game_options {
    uint64_t seed;
    uint8_t has_seed;
    uint8_t has_order;
    uint8_t order[4];
    uint8_t shared_objectives_enabled;
    uint8_t pencil_powers_enabled;
    uint8_t objective_count;
    uint8_t objective_cards[2];
    uint8_t has_power_assignments;
    int8_t power_assignments[4];
} ns_game_options;

#pragma pack(pop)

typedef void* ns_game_handle;
typedef void* ns_expansion_handle;

/*
 * Query mode uses null output buffers and zero capacities. The required
 * counts are always written before capacity is checked.
 */
NS_ENGINE_API int ns_expand_afterstate(
    const ns_public_state* input,
    ns_outcome* outcomes,
    int32_t outcome_capacity,
    ns_candidate* candidates,
    int32_t candidate_capacity,
    int32_t* outcome_count,
    int32_t* candidate_count);

NS_ENGINE_API int ns_analyze_afterstate(
    const ns_public_state* input,
    ns_state_metrics* metrics);

/* Encode canonical afterstates into row-major float32 feature vectors. */
NS_ENGINE_API int ns_feature_afterstates(
    const ns_public_state* inputs,
    int32_t input_count,
    float* features,
    int32_t feature_row_capacity);

/* Build one immutable, owner-ordered exact-target expansion. */
NS_ENGINE_API int ns_expansion_create(
    const ns_public_state* inputs,
    int32_t input_count,
    ns_expansion_handle* destination);
NS_ENGINE_API void ns_expansion_destroy(ns_expansion_handle expansion);
NS_ENGINE_API int32_t ns_expansion_thread_count(int32_t input_count);
NS_ENGINE_API int32_t ns_expansion_outcome_count(ns_expansion_handle expansion);
NS_ENGINE_API int32_t ns_expansion_candidate_count(ns_expansion_handle expansion);
NS_ENGINE_API const int32_t* ns_expansion_outcome_owners(
    ns_expansion_handle expansion);
NS_ENGINE_API const double* ns_expansion_outcome_probabilities(
    ns_expansion_handle expansion);
NS_ENGINE_API const int32_t* ns_expansion_outcome_candidate_offsets(
    ns_expansion_handle expansion);
NS_ENGINE_API const int32_t* ns_expansion_outcome_candidate_counts(
    ns_expansion_handle expansion);
NS_ENGINE_API const int32_t* ns_expansion_candidate_rewards(
    ns_expansion_handle expansion);
NS_ENGINE_API const uint8_t* ns_expansion_candidate_terminated(
    ns_expansion_handle expansion);
/* Row-major candidate_count x ns_observation_dim(). */
NS_ENGINE_API const float* ns_expansion_candidate_features(
    ns_expansion_handle expansion);
/* Select the first maximum candidate for every outcome. */
NS_ENGINE_API int ns_expansion_select_candidates(
    ns_expansion_handle expansion,
    const float* online_values,
    int32_t online_value_count,
    double reward_scale,
    double gamma,
    int64_t* selected_indices,
    int32_t selected_index_count);
/* Reduce selected target values to one float32 target per replay owner. */
NS_ENGINE_API int ns_expansion_reduce_targets(
    ns_expansion_handle expansion,
    const int64_t* selected_indices,
    int32_t selected_index_count,
    const float* target_values,
    int32_t target_value_count,
    double reward_scale,
    double gamma,
    float* targets,
    int32_t target_count);

NS_ENGINE_API int ns_game_create(
    const uint8_t order[4],
    ns_game_handle* destination);
NS_ENGINE_API int ns_game_create_configured(
    const ns_game_options* options,
    ns_game_handle* destination);
NS_ENGINE_API int ns_game_create_from_snapshot(
    const ns_game_snapshot* snapshot,
    ns_game_handle* destination);
NS_ENGINE_API int ns_game_clone(
    ns_game_handle source,
    ns_game_handle* destination);
NS_ENGINE_API void ns_game_destroy(ns_game_handle game);
NS_ENGINE_API int ns_game_reset(ns_game_handle game);
NS_ENGINE_API int ns_game_export(
    ns_game_handle game,
    ns_game_snapshot* destination);
NS_ENGINE_API int ns_game_draw_known(
    ns_game_handle game,
    const uint8_t* card_ids,
    int32_t card_count);
NS_ENGINE_API int ns_game_draw(ns_game_handle game);
NS_ENGINE_API int ns_game_legal_actions(
    ns_game_handle game,
    ns_game_action* actions,
    int32_t action_capacity,
    int32_t* action_count);
NS_ENGINE_API int ns_game_apply_action(
    ns_game_handle game,
    const ns_game_action* action);
NS_ENGINE_API int ns_game_serialize(
    ns_game_handle game,
    uint8_t* destination,
    int32_t capacity,
    int32_t* size);
NS_ENGINE_API int ns_game_deserialize(
    const uint8_t* data,
    int32_t size,
    ns_game_handle* destination);

NS_ENGINE_API const char* ns_last_error(void);
NS_ENGINE_API int32_t ns_observation_dim(void);
NS_ENGINE_API int32_t ns_public_state_size(void);
NS_ENGINE_API int32_t ns_outcome_size(void);
NS_ENGINE_API int32_t ns_candidate_size(void);
NS_ENGINE_API int32_t ns_state_metrics_size(void);
NS_ENGINE_API int32_t ns_game_snapshot_size(void);
NS_ENGINE_API int32_t ns_game_action_size(void);
NS_ENGINE_API int32_t ns_game_options_size(void);

#ifdef __cplusplus
}
#endif

#endif  /* NEXT_STATION_NATIVE_C_API_H */
