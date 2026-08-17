#define NEXT_STATION_ENGINE_EXPORTS
#include "next_station/c_api.h"
#include "next_station/engine.hpp"

#include "c_api_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using next_station::native::Candidate;
using next_station::native::ChanceOutcome;
using next_station::native::GameState;
using next_station::native::PublicDrawList;
using next_station::native::PublicState;
using next_station::native::kColorCount;
using next_station::native::kObservationDim;

thread_local std::string g_last_error;

struct NativeGameHandle {
    explicit NativeGameHandle(const GameState& value) : game(value) {}
    GameState game;
};

struct OwnerCounts {
    std::size_t outcomes;
    std::size_t candidates;

    OwnerCounts() : outcomes(0), candidates(0) {}
};

struct NativeExpansionHandle {
    NativeExpansionHandle(
        std::size_t owners,
        std::size_t outcomes,
        std::size_t candidates)
        : owner_count(static_cast<int32_t>(owners)),
          outcome_count(static_cast<int32_t>(outcomes)),
          candidate_count(static_cast<int32_t>(candidates)),
          owner_outcome_offsets(owners == 0 ? 0 : new int32_t[owners]),
          owner_outcome_counts(owners == 0 ? 0 : new int32_t[owners]),
          owner_terminated(owners == 0 ? 0 : new uint8_t[owners]),
          outcome_owners(outcomes == 0 ? 0 : new int32_t[outcomes]),
          outcome_probabilities(outcomes == 0 ? 0 : new double[outcomes]),
          outcome_candidate_offsets(outcomes == 0 ? 0 : new int32_t[outcomes]),
          outcome_candidate_counts(outcomes == 0 ? 0 : new int32_t[outcomes]),
          candidate_rewards(candidates == 0 ? 0 : new int32_t[candidates]),
          candidate_terminated(candidates == 0 ? 0 : new uint8_t[candidates]),
          candidate_features(
              candidates == 0
                  ? 0
                  : new float[candidates * static_cast<std::size_t>(
                        kObservationDim)]) {}

    int32_t owner_count;
    int32_t outcome_count;
    int32_t candidate_count;
    std::unique_ptr<int32_t[]> owner_outcome_offsets;
    std::unique_ptr<int32_t[]> owner_outcome_counts;
    std::unique_ptr<uint8_t[]> owner_terminated;
    std::unique_ptr<int32_t[]> outcome_owners;
    std::unique_ptr<double[]> outcome_probabilities;
    std::unique_ptr<int32_t[]> outcome_candidate_offsets;
    std::unique_ptr<int32_t[]> outcome_candidate_counts;
    std::unique_ptr<int32_t[]> candidate_rewards;
    std::unique_ptr<uint8_t[]> candidate_terminated;
    std::unique_ptr<float[]> candidate_features;
};

unsigned expansion_thread_count(int32_t input_count) {
    if (input_count <= 1) return 1;
    unsigned requested = std::thread::hardware_concurrency();
    if (requested == 0) requested = 1;
    const char* configured = std::getenv("NEXT_STATION_NATIVE_THREADS");
    if (configured != 0 && configured[0] != '\0') {
        char* end = 0;
        const long parsed = std::strtol(configured, &end, 10);
        if (end != configured && *end == '\0' && parsed > 0) {
            requested = static_cast<unsigned>(parsed);
        }
    }
    return std::min(requested, static_cast<unsigned>(input_count));
}

template <typename Function>
void parallel_for(int32_t count, const Function& function) {
    const unsigned worker_count = expansion_thread_count(count);
    if (worker_count <= 1 || count < 4) {
        for (int32_t index = 0; index < count; ++index) function(index);
        return;
    }

    std::atomic<int32_t> next(0);
    std::atomic<bool> failed(false);
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (unsigned worker = 0; worker < worker_count; ++worker) {
        workers.push_back(std::thread([&]() {
            while (!failed.load()) {
                const int32_t index = next.fetch_add(1);
                if (index >= count) return;
                try {
                    function(index);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error) first_error = std::current_exception();
                    }
                    failed.store(true);
                    return;
                }
            }
        }));
    }
    for (std::size_t index = 0; index < workers.size(); ++index) {
        workers[index].join();
    }
    if (first_error) std::rethrow_exception(first_error);
}

NativeGameHandle& require_game(ns_game_handle handle) {
    if (handle == 0) throw std::runtime_error("native game handle is required");
    return *static_cast<NativeGameHandle*>(handle);
}

NativeExpansionHandle& require_expansion(ns_expansion_handle handle) {
    if (handle == 0) {
        throw std::runtime_error("native expansion handle is required");
    }
    return *static_cast<NativeExpansionHandle*>(handle);
}

double candidate_value(
    int32_t reward,
    float continuation,
    double reward_scale,
    double gamma) {
    const double scaled_reward = static_cast<double>(reward) / reward_scale;
    volatile double scaled_continuation =
        static_cast<double>(continuation) * gamma;
    return scaled_reward + scaled_continuation;
}

PublicState from_c_state(const ns_public_state& source) {
    PublicState result;
    for (int color = 0; color < kColorCount; ++color) {
        result.line_station_masks[color] = source.line_station_masks[color];
        for (int word = 0; word < 3; ++word) {
            result.line_edge_masks[color].words[word] =
                source.line_edge_words[color][word];
        }
        result.order[color] = source.order[color];
    }
    result.remaining_mask = source.remaining_mask;
    result.round_index = source.round_index;
    result.underground_count = source.underground_count;
    result.draw_count = source.draw_count;
    result.terminated = source.terminated != 0;
    return result;
}

OwnerCounts count_owner(const ns_public_state& input) {
    OwnerCounts result;
    const GameState game = GameState::from_canonical(from_c_state(input));
    if (game.terminated()) return result;
    const PublicDrawList draws = game.public_draws();
    result.outcomes = draws.count;
    for (std::size_t index = 0; index < draws.count; ++index) {
        const GameState outcome = game.public_successor(draws.items[index]);
        result.candidates += outcome.legal_actions().size() + 1;
    }
    return result;
}

NativeExpansionHandle* build_expansion(
    const ns_public_state* inputs,
    int32_t input_count) {
    std::vector<OwnerCounts> owner_counts(static_cast<std::size_t>(input_count));
    parallel_for(input_count, [&](int32_t owner) {
        owner_counts[static_cast<std::size_t>(owner)] = count_owner(inputs[owner]);
    });

    std::vector<std::size_t> outcome_offsets(
        static_cast<std::size_t>(input_count) + 1, 0);
    std::vector<std::size_t> candidate_offsets(
        static_cast<std::size_t>(input_count) + 1, 0);
    for (int32_t owner = 0; owner < input_count; ++owner) {
        const OwnerCounts& value = owner_counts[static_cast<std::size_t>(owner)];
        outcome_offsets[static_cast<std::size_t>(owner) + 1] =
            outcome_offsets[static_cast<std::size_t>(owner)]
            + value.outcomes;
        candidate_offsets[static_cast<std::size_t>(owner) + 1] =
            candidate_offsets[static_cast<std::size_t>(owner)]
            + value.candidates;
    }
    const std::size_t outcome_count = outcome_offsets.back();
    const std::size_t candidate_count = candidate_offsets.back();
    const std::size_t max_count = static_cast<std::size_t>(
        std::numeric_limits<int32_t>::max());
    if (outcome_count > max_count || candidate_count > max_count) {
        throw std::runtime_error("native expansion exceeds int32 capacity");
    }
    if (candidate_count > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(kObservationDim)) {
        throw std::runtime_error("native expansion feature size overflows");
    }

    NativeExpansionHandle* result = new NativeExpansionHandle(
        static_cast<std::size_t>(input_count), outcome_count, candidate_count);
    try {
        for (int32_t owner = 0; owner < input_count; ++owner) {
            const std::size_t owner_index = static_cast<std::size_t>(owner);
            result->owner_outcome_offsets[owner_index] = static_cast<int32_t>(
                outcome_offsets[owner_index]);
            result->owner_outcome_counts[owner_index] = static_cast<int32_t>(
                outcome_offsets[owner_index + 1] - outcome_offsets[owner_index]);
            result->owner_terminated[owner_index] =
                inputs[owner].terminated != 0 ? 1 : 0;
        }
        parallel_for(input_count, [&](int32_t owner) {
            const std::size_t first_outcome =
                outcome_offsets[static_cast<std::size_t>(owner)];
            const std::size_t first_candidate =
                candidate_offsets[static_cast<std::size_t>(owner)];
            const GameState game = GameState::from_canonical(
                from_c_state(inputs[owner]));
            if (game.terminated()) return;
            const PublicDrawList draws = game.public_draws();
            std::size_t candidate_cursor = first_candidate;
            for (std::size_t index = 0; index < draws.count; ++index) {
                const GameState outcome = game.public_successor(draws.items[index]);
                const std::vector<Candidate> candidates =
                    outcome.candidates();
                const std::size_t outcome_cursor = first_outcome + index;
                if (!std::isfinite(draws.items[index].probability)
                    || draws.items[index].probability <= 0.0
                    || candidates.empty()) {
                    throw std::runtime_error(
                        "native expansion produced an invalid outcome");
                }
                result->outcome_owners[outcome_cursor] = owner;
                result->outcome_probabilities[outcome_cursor] =
                    draws.items[index].probability;
                result->outcome_candidate_offsets[outcome_cursor] =
                    static_cast<int32_t>(candidate_cursor);
                result->outcome_candidate_counts[outcome_cursor] =
                    static_cast<int32_t>(candidates.size());
                for (std::size_t candidate_index = 0;
                     candidate_index < candidates.size(); ++candidate_index) {
                    const Candidate& candidate = candidates[candidate_index];
                    result->candidate_rewards[candidate_cursor] = candidate.reward;
                    result->candidate_terminated[candidate_cursor] =
                        candidate.afterstate.terminated() ? 1 : 0;
                    candidate.afterstate.write_features(
                        result->candidate_features.get()
                            + candidate_cursor * static_cast<std::size_t>(
                                kObservationDim),
                        kObservationDim);
                    ++candidate_cursor;
                }
            }
            if (draws.count
                    != owner_counts[static_cast<std::size_t>(owner)].outcomes
                || candidate_cursor
                    != candidate_offsets[static_cast<std::size_t>(owner) + 1]) {
                throw std::runtime_error(
                    "native owner expansion count changed between passes");
            }
        });
    } catch (...) {
        delete result;
        throw;
    }
    return result;
}

void to_c_state(const PublicState& source, ns_public_state* destination) {
    for (int color = 0; color < kColorCount; ++color) {
        destination->line_station_masks[color] = source.line_station_masks[color];
        for (int word = 0; word < 3; ++word) {
            destination->line_edge_words[color][word] =
                source.line_edge_masks[color].words[word];
        }
        destination->order[color] = source.order[color];
    }
    destination->remaining_mask = source.remaining_mask;
    destination->round_index = source.round_index;
    destination->underground_count = source.underground_count;
    destination->draw_count = source.draw_count;
    destination->terminated = source.terminated ? 1 : 0;
}

PublicState snapshot_public_state(const GameState& game) {
    PublicState result;
    const std::array<next_station::native::LineState, kColorCount>& lines =
        game.lines();
    const std::array<std::uint8_t, kColorCount>& order = game.order();
    for (int color = 0; color < kColorCount; ++color) {
        result.line_station_masks[color] = lines[color].station_mask;
        result.line_edge_masks[color] = lines[color].edge_mask;
        result.order[color] = order[color];
    }
    result.remaining_mask = game.remaining_mask();
    result.round_index = game.round_index();
    result.underground_count = game.underground_count();
    result.draw_count = game.draw_count();
    result.terminated = game.terminated();
    return result;
}

void export_game(const GameState& game, ns_game_snapshot* destination) {
    to_c_state(snapshot_public_state(game), &destination->state);
    const std::array<next_station::native::LineState, kColorCount>& lines =
        game.lines();
    for (int color = 0; color < kColorCount; ++color) {
        destination->line_leaf_masks[color] = lines[color].leaf_mask;
    }
    destination->shared_objectives_enabled =
        game.shared_objectives_enabled() ? 1 : 0;
    destination->pencil_powers_enabled = game.pencil_powers_enabled() ? 1 : 0;
    destination->objective_cards[0] = game.objective_cards()[0];
    destination->objective_cards[1] = game.objective_cards()[1];
    destination->shared_objective_mask = game.shared_objective_mask();
    for (int color = 0; color < kColorCount; ++color) {
        destination->power_assignments[color] = game.power_assignments()[color];
    }
    destination->used_power_mask = game.used_power_mask();
    destination->completed_objective_mask = game.completed_objective_mask();
    destination->double_section_pending =
        game.double_section_pending() ? 1 : 0;
    destination->double_target_symbol = static_cast<int8_t>(
        game.double_target_symbol());
    destination->has_pending = game.has_pending() ? 1 : 0;
    destination->pending_card_ids[0] = 0;
    destination->pending_card_ids[1] = 0;
    destination->pending_card_count = 0;
    destination->pending_target_symbol = -1;
    destination->pending_wild = 0;
    destination->pending_source_any = 0;
    destination->pending_final_card = 0;
    if (game.has_pending()) {
        const next_station::native::PendingEvent& event = game.pending();
        destination->pending_card_ids[0] = event.card_ids[0];
        destination->pending_card_ids[1] = event.card_ids[1];
        destination->pending_card_count = event.count;
        destination->pending_target_symbol =
            static_cast<int8_t>(event.target_symbol);
        destination->pending_wild = event.wild ? 1 : 0;
        destination->pending_source_any = event.source_any ? 1 : 0;
        destination->pending_final_card = event.final_card ? 1 : 0;
    }
    const std::array<int, 6> partial = game.partial_score_components();
    for (int index = 0; index < 6; ++index) {
        destination->partial_components[index] = partial[index];
    }
    const next_station::native::FinalScore score = game.final_score();
    destination->score_summary[0] = score.line_total;
    destination->score_summary[1] = score.tourist_visits;
    destination->score_summary[2] = score.tourist_bonus;
    destination->score_summary[3] = score.two_line_stations;
    destination->score_summary[4] = score.three_line_stations;
    destination->score_summary[5] = score.four_line_stations;
    destination->score_summary[6] = score.interchange_bonus;
    destination->score_summary[7] = score.objectives_completed;
    destination->score_summary[8] = score.objective_bonus;
    destination->score_summary[9] = score.total;
    const std::vector<next_station::native::LineScore>& round_scores =
        game.round_scores();
    destination->round_score_count = static_cast<uint8_t>(round_scores.size());
    std::memset(destination->round_scores, 0, sizeof(destination->round_scores));
    for (std::size_t index = 0; index < round_scores.size(); ++index) {
        destination->round_scores[index][0] = round_scores[index].districts;
        destination->round_scores[index][1] = round_scores[index].max_stations;
        destination->round_scores[index][2] = round_scores[index].thames_crossings;
        destination->round_scores[index][3] = round_scores[index].route;
        destination->round_scores[index][4] = round_scores[index].thames;
        destination->round_scores[index][5] = round_scores[index].total;
    }
}

GameState import_game(const ns_game_snapshot& source) {
    GameState game = GameState::from_canonical(from_c_state(source.state));
    std::array<std::uint8_t, 2> objectives = {{
        source.objective_cards[0], source.objective_cards[1],
    }};
    std::array<std::int8_t, kColorCount> powers;
    for (int color = 0; color < kColorCount; ++color) {
        powers[color] = source.power_assignments[color];
    }
    game.configure_advanced(
        source.shared_objectives_enabled != 0,
        objectives,
        source.pencil_powers_enabled != 0,
        powers);
    game.restore_advanced_state(
        source.used_power_mask,
        source.completed_objective_mask,
        source.double_section_pending != 0,
        static_cast<next_station::native::Symbol>(source.double_target_symbol));
    if (source.has_pending) {
        next_station::native::PendingEvent event;
        event.card_ids[0] = source.pending_card_ids[0];
        event.card_ids[1] = source.pending_card_ids[1];
        event.count = source.pending_card_count;
        event.target_symbol = static_cast<next_station::native::Symbol>(
            source.pending_target_symbol);
        event.wild = source.pending_wild != 0;
        event.source_any = source.pending_source_any != 0;
        event.final_card = source.pending_final_card != 0;
        game.restore_pending(event);
    }
    return game;
}

std::uint64_t random_seed() {
    std::random_device device;
    const std::uint64_t high = static_cast<std::uint64_t>(device());
    const std::uint64_t low = static_cast<std::uint64_t>(device());
    return (high << 32) ^ low;
}

next_station::native::GameOptions from_c_options(const ns_game_options& source) {
    next_station::native::GameOptions result;
    result.seed = source.has_seed ? source.seed : random_seed();
    result.has_order = source.has_order != 0;
    for (int color = 0; color < kColorCount; ++color) {
        result.order[color] = source.order[color];
    }
    result.shared_objectives_enabled = source.shared_objectives_enabled != 0;
    result.pencil_powers_enabled = source.pencil_powers_enabled != 0;
    result.objective_count = source.objective_count;
    result.objective_cards[0] = source.objective_cards[0];
    result.objective_cards[1] = source.objective_cards[1];
    result.has_power_assignments = source.has_power_assignments != 0;
    for (int color = 0; color < kColorCount; ++color) {
        result.power_assignments[color] = source.power_assignments[color];
    }
    return result;
}

void append_u32(std::vector<std::uint8_t>* destination, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        destination->push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

std::uint32_t read_u32(
    const std::uint8_t* data,
    std::size_t size,
    std::size_t* cursor) {
    if (*cursor + 4 > size) throw std::runtime_error("serialized game is truncated");
    std::uint32_t result = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        result |= static_cast<std::uint32_t>(data[(*cursor)++]) << shift;
    }
    return result;
}

std::vector<std::uint8_t> serialize_game(const GameState& game) {
    static const char magic[8] = {'N', 'S', 'G', 'A', 'M', 'E', '0', '1'};
    ns_game_snapshot snapshot;
    export_game(game, &snapshot);
    const std::string random = game.random_state();
    const std::vector<std::uint8_t>& deck_order = game.hidden_deck_order();
    std::vector<std::uint8_t> result;
    result.insert(result.end(), magic, magic + sizeof(magic));
    append_u32(&result, static_cast<std::uint32_t>(sizeof(snapshot)));
    const std::uint8_t* snapshot_bytes =
        reinterpret_cast<const std::uint8_t*>(&snapshot);
    result.insert(result.end(), snapshot_bytes, snapshot_bytes + sizeof(snapshot));
    result.push_back(game.is_public_copy() ? 1 : 0);
    result.push_back(static_cast<std::uint8_t>(deck_order.size()));
    result.insert(result.end(), deck_order.begin(), deck_order.end());
    append_u32(&result, static_cast<std::uint32_t>(random.size()));
    result.insert(result.end(), random.begin(), random.end());
    return result;
}

GameState deserialize_game(const std::uint8_t* data, std::size_t size) {
    static const char magic[8] = {'N', 'S', 'G', 'A', 'M', 'E', '0', '1'};
    if (data == 0 || size < sizeof(magic)
        || std::memcmp(data, magic, sizeof(magic)) != 0) {
        throw std::runtime_error("serialized game has an invalid header");
    }
    std::size_t cursor = sizeof(magic);
    const std::uint32_t snapshot_size = read_u32(data, size, &cursor);
    if (snapshot_size != sizeof(ns_game_snapshot)
        || cursor + snapshot_size > size) {
        throw std::runtime_error("serialized game snapshot is incompatible");
    }
    ns_game_snapshot snapshot;
    std::memcpy(&snapshot, data + cursor, sizeof(snapshot));
    cursor += sizeof(snapshot);
    if (cursor + 2 > size) throw std::runtime_error("serialized game is truncated");
    const bool public_copy = data[cursor++] != 0;
    const std::size_t deck_count = data[cursor++];
    if (cursor + deck_count > size) {
        throw std::runtime_error("serialized game deck is truncated");
    }
    std::vector<std::uint8_t> deck_order(
        data + cursor, data + cursor + deck_count);
    cursor += deck_count;
    const std::uint32_t random_size = read_u32(data, size, &cursor);
    if (cursor + random_size != size) {
        throw std::runtime_error("serialized game random state is truncated");
    }
    const std::string random(
        reinterpret_cast<const char*>(data + cursor), random_size);
    GameState game = import_game(snapshot);
    game.restore_hidden_state(random, deck_order, public_copy);
    return game;
}

}  // namespace

namespace next_station {
namespace native {

GameState& game_from_c_handle(ns_game_handle handle) {
    return ::require_game(handle).game;
}

}  // namespace native
}  // namespace next_station

extern "C" int ns_expand_afterstate(
    const ns_public_state* input,
    ns_outcome* outcomes,
    int32_t outcome_capacity,
    ns_candidate* candidate_buffer,
    int32_t candidate_capacity,
    int32_t* outcome_count,
    int32_t* candidate_count) {
    try {
        g_last_error.clear();
        if (input == 0 || outcome_count == 0 || candidate_count == 0) {
            throw std::runtime_error("input and count pointers are required");
        }
        if (outcome_capacity < 0 || candidate_capacity < 0) {
            throw std::runtime_error("buffer capacities cannot be negative");
        }
        const GameState game = GameState::from_canonical(from_c_state(*input));
        std::vector<ChanceOutcome> expanded;
        if (!game.terminated()) expanded = game.public_successors();
        std::vector<std::vector<Candidate> > groups;
        groups.reserve(expanded.size());
        int32_t total_candidates = 0;
        for (std::size_t i = 0; i < expanded.size(); ++i) {
            groups.push_back(expanded[i].state.candidates());
            total_candidates += static_cast<int32_t>(groups.back().size());
        }
        *outcome_count = static_cast<int32_t>(expanded.size());
        *candidate_count = total_candidates;
        if (outcomes == 0 || candidate_buffer == 0) return 0;
        if (outcome_capacity < *outcome_count
            || candidate_capacity < *candidate_count) {
            throw std::runtime_error("output buffer capacity is too small");
        }
        int32_t cursor = 0;
        for (std::size_t i = 0; i < expanded.size(); ++i) {
            const next_station::native::PendingEvent& event =
                expanded[i].state.pending();
            outcomes[i].probability = expanded[i].probability;
            outcomes[i].candidate_offset = cursor;
            outcomes[i].candidate_count = static_cast<int32_t>(groups[i].size());
            outcomes[i].card_ids[0] = event.card_ids[0];
            outcomes[i].card_ids[1] = event.card_ids[1];
            outcomes[i].card_count = event.count;
            for (std::size_t j = 0; j < groups[i].size(); ++j) {
                const Candidate& candidate = groups[i][j];
                ns_candidate& destination = candidate_buffer[cursor++];
                destination.action_index = candidate.action_index;
                destination.source = candidate.action.source;
                destination.target = candidate.action.target;
                destination.reward = candidate.reward;
                to_c_state(candidate.afterstate.canonical(), &destination.afterstate);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_analyze_afterstate(
    const ns_public_state* input,
    ns_state_metrics* destination) {
    try {
        g_last_error.clear();
        if (input == 0 || destination == 0) {
            throw std::runtime_error("input and metrics pointers are required");
        }
        const GameState game = GameState::from_canonical(from_c_state(*input));
        const std::array<next_station::native::LineMetrics, kColorCount>& lines =
            game.line_metrics();
        for (int color = 0; color < kColorCount; ++color) {
            destination->line_district_masks[color] = lines[color].district_mask;
            destination->line_district_counts[color] = 0;
            std::uint16_t mask = lines[color].district_mask;
            while (mask != 0) {
                mask = static_cast<std::uint16_t>(mask & (mask - 1));
                ++destination->line_district_counts[color];
            }
            destination->line_max_stations[color] = lines[color].max_stations;
            destination->line_routes[color] = lines[color].route;
            destination->line_thames_crossings[color] =
                lines[color].thames_crossings;
            destination->line_tourist_visits[color] = lines[color].tourist_visits;
            for (int district = 0; district < next_station::native::kDistrictCount;
                 ++district) {
                destination->line_station_counts[color][district] =
                    lines[color].station_counts[district];
            }
        }
        const std::array<int, 6> partial = game.partial_score_components();
        for (int index = 0; index < 6; ++index) {
            destination->partial_components[index] = partial[index];
        }
        const std::array<std::uint8_t, kColorCount + 1>& interchanges =
            game.interchange_counts();
        for (int count = 0; count <= kColorCount; ++count) {
            destination->interchange_counts[count] = interchanges[count];
        }
        destination->current_total = game.current_total();
        destination->final_total = game.terminated()
            ? game.final_score().total : -1;
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int32_t ns_station_count(void) {
    return next_station::native::kStationCount;
}

extern "C" int32_t ns_edge_count(void) {
    return next_station::native::kEdgeCount;
}

extern "C" int32_t ns_card_count(void) {
    return next_station::native::kCardCount;
}

extern "C" int32_t ns_district_count(void) {
    return next_station::native::kDistrictCount;
}

extern "C" int ns_station_get(int32_t id, ns_station_info* destination) {
    try {
        g_last_error.clear();
        const next_station::native::Map& map = next_station::native::london_map();
        if (destination == 0 || id < 0
            || id >= static_cast<int32_t>(map.stations.size())) {
            throw std::runtime_error("station metadata query is invalid");
        }
        const next_station::native::Station& station = map.stations[id];
        destination->id = station.id;
        destination->x = station.x;
        destination->y = station.y;
        destination->symbol = static_cast<int8_t>(station.symbol);
        destination->district = static_cast<int8_t>(station.district);
        destination->tourist = station.tourist ? 1 : 0;
        destination->departure_color = static_cast<int8_t>(
            station.departure_color);
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_edge_get(int32_t id, ns_edge_info* destination) {
    try {
        g_last_error.clear();
        const next_station::native::Map& map = next_station::native::london_map();
        if (destination == 0 || id < 0
            || id >= static_cast<int32_t>(map.edges.size())) {
            throw std::runtime_error("edge metadata query is invalid");
        }
        const next_station::native::Edge& edge = map.edges[id];
        destination->id = edge.id;
        destination->u = edge.u;
        destination->v = edge.v;
        destination->crosses_thames = edge.crosses_thames ? 1 : 0;
        destination->district_mask = edge.district_mask;
        for (int word = 0; word < 3; ++word) {
            destination->conflict_words[word] =
                map.conflict_masks[id].words[word];
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_card_get(int32_t id, ns_card_info* destination) {
    try {
        g_last_error.clear();
        const std::array<next_station::native::Card,
                         next_station::native::kCardCount>& cards =
            next_station::native::deck();
        if (destination == 0 || id < 0
            || id >= static_cast<int32_t>(cards.size())) {
            throw std::runtime_error("card metadata query is invalid");
        }
        const next_station::native::Card& card = cards[id];
        destination->id = card.id;
        destination->symbol = static_cast<int8_t>(card.symbol);
        destination->underground = card.underground ? 1 : 0;
        destination->is_switch = card.is_switch ? 1 : 0;
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" const char* ns_district_name(int32_t id) {
    try {
        g_last_error.clear();
        const next_station::native::Map& map = next_station::native::london_map();
        if (id < 0 || id >= static_cast<int32_t>(map.district_names.size())) {
            throw std::runtime_error("district metadata query is invalid");
        }
        return map.district_names[id].c_str();
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 0;
    }
}

extern "C" int ns_legal_edge_mask(
    const ns_public_state* input,
    int8_t target_symbol,
    uint8_t wild,
    uint8_t source_any,
    uint64_t destination[3]) {
    try {
        g_last_error.clear();
        if (input == 0 || destination == 0) {
            throw std::runtime_error(
                "legal-mask input and destination pointers are required");
        }
        const GameState game = GameState::from_canonical(from_c_state(*input));
        const std::vector<next_station::native::Action> actions =
            game.legal_actions_for_event(
                static_cast<next_station::native::Symbol>(target_symbol),
                wild != 0,
                source_any != 0);
        destination[0] = 0;
        destination[1] = 0;
        destination[2] = 0;
        for (std::size_t index = 0; index < actions.size(); ++index) {
            const int edge = actions[index].edge_id;
            destination[edge / 64] |= uint64_t(1) << (edge % 64);
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_feature_afterstates(
    const ns_public_state* inputs,
    int32_t input_count,
    float* features,
    int32_t feature_row_capacity) {
    try {
        g_last_error.clear();
        if (input_count < 0 || feature_row_capacity < 0) {
            throw std::runtime_error("feature batch sizes cannot be negative");
        }
        if (input_count > 0 && (inputs == 0 || features == 0)) {
            throw std::runtime_error("feature input and output pointers are required");
        }
        if (feature_row_capacity < input_count) {
            throw std::runtime_error("feature output capacity is too small");
        }
        for (int32_t index = 0; index < input_count; ++index) {
            const GameState game = GameState::from_canonical(from_c_state(inputs[index]));
            game.write_features(
                features + static_cast<std::size_t>(index) * kObservationDim,
                kObservationDim);
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_expansion_create(
    const ns_public_state* inputs,
    int32_t input_count,
    ns_expansion_handle* destination) {
    try {
        g_last_error.clear();
        if (input_count < 0) {
            throw std::runtime_error("expansion input count cannot be negative");
        }
        if (destination == 0 || (input_count > 0 && inputs == 0)) {
            throw std::runtime_error(
                "expansion input and destination pointers are required");
        }
        *destination = 0;
        *destination = static_cast<ns_expansion_handle>(
            build_expansion(inputs, input_count));
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" void ns_expansion_destroy(ns_expansion_handle expansion) {
    delete static_cast<NativeExpansionHandle*>(expansion);
}

extern "C" int32_t ns_expansion_thread_count(int32_t input_count) {
    if (input_count < 1) return 0;
    return static_cast<int32_t>(expansion_thread_count(input_count));
}

extern "C" int32_t ns_expansion_outcome_count(ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 ? 0 : value->outcome_count;
}

extern "C" int32_t ns_expansion_candidate_count(ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 ? 0 : value->candidate_count;
}

extern "C" const int32_t* ns_expansion_outcome_owners(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->outcome_count == 0
        ? 0 : value->outcome_owners.get();
}

extern "C" const double* ns_expansion_outcome_probabilities(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->outcome_count == 0
        ? 0 : value->outcome_probabilities.get();
}

extern "C" const int32_t* ns_expansion_outcome_candidate_offsets(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->outcome_count == 0
        ? 0 : value->outcome_candidate_offsets.get();
}

extern "C" const int32_t* ns_expansion_outcome_candidate_counts(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->outcome_count == 0
        ? 0 : value->outcome_candidate_counts.get();
}

extern "C" const int32_t* ns_expansion_candidate_rewards(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->candidate_count == 0
        ? 0 : value->candidate_rewards.get();
}

extern "C" const uint8_t* ns_expansion_candidate_terminated(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->candidate_count == 0
        ? 0 : value->candidate_terminated.get();
}

extern "C" const float* ns_expansion_candidate_features(
    ns_expansion_handle expansion) {
    const NativeExpansionHandle* value =
        static_cast<const NativeExpansionHandle*>(expansion);
    return value == 0 || value->candidate_count == 0
        ? 0 : value->candidate_features.get();
}

extern "C" int ns_expansion_select_candidates(
    ns_expansion_handle expansion,
    const float* online_values,
    int32_t online_value_count,
    double reward_scale,
    double gamma,
    int64_t* selected_indices,
    int32_t selected_index_count) {
    try {
        g_last_error.clear();
        NativeExpansionHandle& value = require_expansion(expansion);
        if (!std::isfinite(reward_scale) || reward_scale <= 0.0
            || !std::isfinite(gamma) || gamma <= 0.0 || gamma > 1.0) {
            throw std::runtime_error(
                "native selection scale and gamma are invalid");
        }
        if (online_value_count != value.candidate_count
            || selected_index_count != value.outcome_count) {
            throw std::runtime_error(
                "native selection array lengths are incompatible");
        }
        if ((online_value_count > 0 && online_values == 0)
            || (selected_index_count > 0 && selected_indices == 0)) {
            throw std::runtime_error(
                "native selection array pointers are required");
        }
        for (int32_t outcome = 0; outcome < value.outcome_count; ++outcome) {
            const int32_t start = value.outcome_candidate_offsets[outcome];
            const int32_t count = value.outcome_candidate_counts[outcome];
            if (start < 0 || count < 1
                || start > value.candidate_count - count) {
                throw std::runtime_error(
                    "native selection candidate range is invalid");
            }
            int32_t best = start;
            double best_score = candidate_value(
                value.candidate_rewards[start],
                online_values[start],
                reward_scale,
                gamma);
            if (!std::isnan(best_score)) {
                for (int32_t candidate = start + 1;
                     candidate < start + count; ++candidate) {
                    const double score = candidate_value(
                        value.candidate_rewards[candidate],
                        online_values[candidate],
                        reward_scale,
                        gamma);
                    if (std::isnan(score)) {
                        best = candidate;
                        break;
                    }
                    if (score > best_score) {
                        best = candidate;
                        best_score = score;
                    }
                }
            }
            selected_indices[outcome] = static_cast<int64_t>(best);
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_expansion_reduce_targets(
    ns_expansion_handle expansion,
    const int64_t* selected_indices,
    int32_t selected_index_count,
    const float* target_values,
    int32_t target_value_count,
    double reward_scale,
    double gamma,
    float* targets,
    int32_t target_count) {
    try {
        g_last_error.clear();
        NativeExpansionHandle& value = require_expansion(expansion);
        if (!std::isfinite(reward_scale) || reward_scale <= 0.0
            || !std::isfinite(gamma) || gamma <= 0.0 || gamma > 1.0) {
            throw std::runtime_error(
                "native reduction scale and gamma are invalid");
        }
        if (selected_index_count != value.outcome_count
            || target_value_count != value.outcome_count
            || target_count != value.owner_count) {
            throw std::runtime_error(
                "native reduction array lengths are incompatible");
        }
        if ((selected_index_count > 0 && selected_indices == 0)
            || (target_value_count > 0 && target_values == 0)
            || (target_count > 0 && targets == 0)) {
            throw std::runtime_error(
                "native reduction array pointers are required");
        }
        if (target_count > 0) {
            std::fill(targets, targets + target_count, 0.0f);
        }
        for (int32_t owner = 0; owner < value.owner_count; ++owner) {
            const int32_t first = value.owner_outcome_offsets[owner];
            const int32_t count = value.owner_outcome_counts[owner];
            if (first < 0 || count < 0
                || first > value.outcome_count - count) {
                throw std::runtime_error(
                    "native reduction outcome range is invalid");
            }
            if (count == 0) {
                if (value.owner_terminated[owner] == 0) {
                    throw std::runtime_error(
                        "nonterminal afterstate has no chance outcomes");
                }
                continue;
            }
            if (value.owner_terminated[owner] != 0) {
                throw std::runtime_error(
                    "terminal afterstate has chance outcomes");
            }
            double total = 0.0;
            double probability_sum = 0.0;
            for (int32_t outcome = first; outcome < first + count; ++outcome) {
                const int32_t candidate = static_cast<int32_t>(
                    selected_indices[outcome]);
                const int32_t candidate_start =
                    value.outcome_candidate_offsets[outcome];
                const int32_t candidate_count =
                    value.outcome_candidate_counts[outcome];
                if (selected_indices[outcome] < candidate_start
                    || selected_indices[outcome]
                        >= static_cast<int64_t>(candidate_start)
                            + candidate_count) {
                    throw std::runtime_error(
                        "native reduction selected index is invalid");
                }
                const double probability =
                    value.outcome_probabilities[outcome];
                probability_sum += probability;
                const double selected_value = candidate_value(
                    value.candidate_rewards[candidate],
                    target_values[outcome],
                    reward_scale,
                    gamma);
                volatile double weighted_value = probability * selected_value;
                total += weighted_value;
            }
            const double probability_tolerance = 1e-12 + 1e-5;
            if (!std::isfinite(probability_sum)
                || std::fabs(probability_sum - 1.0) > probability_tolerance) {
                throw std::runtime_error(
                    "native chance probabilities do not sum to one");
            }
            targets[owner] = static_cast<float>(total);
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_create(
    const uint8_t order[4],
    ns_game_handle* destination) {
    try {
        g_last_error.clear();
        if (order == 0 || destination == 0) {
            throw std::runtime_error("order and destination pointers are required");
        }
        std::array<std::uint8_t, kColorCount> native_order;
        for (int index = 0; index < kColorCount; ++index) {
            native_order[index] = order[index];
        }
        *destination = new NativeGameHandle(GameState(native_order, 0));
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_create_configured(
    const ns_game_options* options,
    ns_game_handle* destination) {
    try {
        g_last_error.clear();
        if (options == 0 || destination == 0) {
            throw std::runtime_error("game options and destination are required");
        }
        *destination = new NativeGameHandle(GameState(from_c_options(*options)));
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_create_from_snapshot(
    const ns_game_snapshot* snapshot,
    ns_game_handle* destination) {
    try {
        g_last_error.clear();
        if (snapshot == 0 || destination == 0) {
            throw std::runtime_error("snapshot and destination pointers are required");
        }
        *destination = new NativeGameHandle(import_game(*snapshot));
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_clone(
    ns_game_handle source,
    ns_game_handle* destination) {
    try {
        g_last_error.clear();
        if (destination == 0) {
            throw std::runtime_error("clone destination pointer is required");
        }
        *destination = new NativeGameHandle(require_game(source).game.copy_public());
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" void ns_game_destroy(ns_game_handle game) {
    delete static_cast<NativeGameHandle*>(game);
}

extern "C" int ns_game_reset(ns_game_handle handle) {
    try {
        g_last_error.clear();
        require_game(handle).game.reset();
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_export(
    ns_game_handle handle,
    ns_game_snapshot* destination) {
    try {
        g_last_error.clear();
        if (destination == 0) {
            throw std::runtime_error("snapshot destination pointer is required");
        }
        export_game(require_game(handle).game, destination);
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_draw_known(
    ns_game_handle handle,
    const uint8_t* card_ids,
    int32_t card_count) {
    try {
        g_last_error.clear();
        if (card_count < 1 || card_ids == 0) {
            throw std::runtime_error("known draw requires card ids");
        }
        std::vector<int> cards;
        cards.reserve(card_count);
        for (int32_t index = 0; index < card_count; ++index) {
            cards.push_back(card_ids[index]);
        }
        require_game(handle).game.draw_known_cards(cards);
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_draw(ns_game_handle handle) {
    try {
        g_last_error.clear();
        require_game(handle).game.draw();
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_legal_actions(
    ns_game_handle handle,
    ns_game_action* actions,
    int32_t action_capacity,
    int32_t* action_count) {
    try {
        g_last_error.clear();
        if (action_capacity < 0 || action_count == 0) {
            throw std::runtime_error("action capacity and count are invalid");
        }
        const GameState& game = require_game(handle).game;
        const std::vector<next_station::native::Action> legal = game.legal_actions();
        *action_count = static_cast<int32_t>(legal.size());
        if (actions == 0) return 0;
        if (action_capacity < *action_count) {
            throw std::runtime_error("action output capacity is too small");
        }
        for (std::size_t index = 0; index < legal.size(); ++index) {
            const next_station::native::Action& action = legal[index];
            const next_station::native::ScoreDelta reward =
                game.score_delta_for_legal_action(action);
            actions[index].edge_id = action.edge_id;
            actions[index].source = action.source;
            actions[index].target = action.target;
            actions[index].power = static_cast<int8_t>(action.power);
            actions[index].reward_components[0] = reward.route;
            actions[index].reward_components[1] = reward.thames;
            actions[index].reward_components[2] = reward.tourist;
            actions[index].reward_components[3] = reward.interchange;
            actions[index].reward_components[4] = reward.objective;
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_apply_action(
    ns_game_handle handle,
    const ns_game_action* action) {
    try {
        g_last_error.clear();
        GameState& game = require_game(handle).game;
        if (action == 0) {
            game.apply_action(next_station::native::Action());
        } else {
            game.apply_action(next_station::native::Action(
                action->edge_id, action->source, action->target,
                static_cast<next_station::native::PencilPower>(action->power)));
        }
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_serialize(
    ns_game_handle handle,
    uint8_t* destination,
    int32_t capacity,
    int32_t* size) {
    try {
        g_last_error.clear();
        if (capacity < 0 || size == 0) {
            throw std::runtime_error("serialization capacity and size are invalid");
        }
        const std::vector<std::uint8_t> data =
            serialize_game(require_game(handle).game);
        if (data.size() > static_cast<std::size_t>(INT32_MAX)) {
            throw std::runtime_error("serialized game is too large");
        }
        *size = static_cast<int32_t>(data.size());
        if (destination == 0) return 0;
        if (capacity < *size) {
            throw std::runtime_error("serialization buffer is too small");
        }
        std::memcpy(destination, data.data(), data.size());
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" int ns_game_deserialize(
    const uint8_t* data,
    int32_t size,
    ns_game_handle* destination) {
    try {
        g_last_error.clear();
        if (data == 0 || size < 0 || destination == 0) {
            throw std::runtime_error("serialized game and destination are required");
        }
        *destination = new NativeGameHandle(
            deserialize_game(data, static_cast<std::size_t>(size)));
        return 0;
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return 1;
    } catch (...) {
        g_last_error = "unknown native engine error";
        return 1;
    }
}

extern "C" const char* ns_last_error(void) {
    return g_last_error.c_str();
}

extern "C" int32_t ns_observation_dim(void) {
    return kObservationDim;
}

extern "C" int32_t ns_public_state_size(void) {
    return static_cast<int32_t>(sizeof(ns_public_state));
}

extern "C" int32_t ns_outcome_size(void) {
    return static_cast<int32_t>(sizeof(ns_outcome));
}

extern "C" int32_t ns_candidate_size(void) {
    return static_cast<int32_t>(sizeof(ns_candidate));
}

extern "C" int32_t ns_state_metrics_size(void) {
    return static_cast<int32_t>(sizeof(ns_state_metrics));
}

extern "C" int32_t ns_station_info_size(void) {
    return static_cast<int32_t>(sizeof(ns_station_info));
}

extern "C" int32_t ns_edge_info_size(void) {
    return static_cast<int32_t>(sizeof(ns_edge_info));
}

extern "C" int32_t ns_card_info_size(void) {
    return static_cast<int32_t>(sizeof(ns_card_info));
}

extern "C" int32_t ns_game_snapshot_size(void) {
    return static_cast<int32_t>(sizeof(ns_game_snapshot));
}

extern "C" int32_t ns_game_action_size(void) {
    return static_cast<int32_t>(sizeof(ns_game_action));
}

extern "C" int32_t ns_game_options_size(void) {
    return static_cast<int32_t>(sizeof(ns_game_options));
}
