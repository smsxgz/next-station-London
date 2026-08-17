#ifndef NEXT_STATION_NATIVE_ENGINE_HPP
#define NEXT_STATION_NATIVE_ENGINE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace next_station {
namespace native {

static const int kColorCount = 4;
static const int kStationCount = 53;
static const int kEdgeCount = 155;
static const int kCardCount = 11;
static const int kDistrictCount = 13;
static const int kPassAction = kEdgeCount;
static const int kObservationDim = 1041;
static const int kObjectiveCount = 5;
static const int kPowerCount = 4;
static const int kMaxPublicDrawCount = 2 * (kCardCount - 1);

enum Color : std::uint8_t {
    kPurple = 0,
    kBlue = 1,
    kPink = 2,
    kGreen = 3,
};

enum Symbol : std::int8_t {
    kCircle = 0,
    kTriangle = 1,
    kSquare = 2,
    kPentagon = 3,
    kCentral = 4,
    kWild = -1,
};

enum class Status : std::uint8_t {
    Playing = 0,
    Finished = 1,
};

enum Objective : std::uint8_t {
    kEightInterchanges = 0,
    kAllDistricts = 1,
    kAllTouristSites = 2,
    kAllCentralStations = 3,
    kSixThamesCrossings = 4,
};

enum PencilPower : std::int8_t {
    kNoPower = -1,
    kDoubleSection = 0,
    kWildCard = 1,
    kRailroadSwitch = 2,
    kCircleStation = 3,
};

/* A small fixed-width bitset for the 155 possible map sections. */
struct Mask155 {
    std::uint64_t words[3];

    Mask155();
    explicit Mask155(std::uint64_t first_word);

    void clear();
    bool test(int index) const;
    void set(int index);
    void reset(int index);
    bool intersects(const Mask155& other) const;
    int count() const;
    Mask155& operator|=(const Mask155& other);
    Mask155& operator&=(const Mask155& other);
};

bool operator==(const Mask155& left, const Mask155& right);
bool operator!=(const Mask155& left, const Mask155& right);
Mask155 operator|(Mask155 left, const Mask155& right);
Mask155 operator&(Mask155 left, const Mask155& right);

struct Station {
    int id;
    int x;
    int y;
    Symbol symbol;
    int district;
    bool tourist;
    int departure_color;
};

struct Edge {
    int id;
    int u;
    int v;
    bool crosses_thames;
    std::uint16_t district_mask;
};

struct Map {
    std::vector<Station> stations;
    std::vector<Edge> edges;
    std::vector<std::string> district_names;
    std::vector<std::vector<int> > adjacency;
    std::vector<std::vector<std::pair<int, int> > > oriented_adjacency;
    std::vector<Mask155> conflict_masks;
    std::array<int, kStationCount> station_district_indices;
    std::array<std::uint16_t, kEdgeCount> edge_district_masks;
    int district_count;
};

const Map& london_map();

struct Card {
    int id;
    Symbol symbol;
    bool underground;
    bool is_switch;
};

const std::array<Card, kCardCount>& deck();

struct Action {
    int edge_id;
    int source;
    int target;
    PencilPower power;

    Action();
    Action(int edge, int from, int to, PencilPower selected_power = kNoPower);
    bool is_pass() const;
};

bool operator==(const Action& left, const Action& right);

struct PendingEvent {
    std::array<std::uint8_t, 2> card_ids;
    std::uint8_t count;
    Symbol target_symbol;
    bool wild;
    bool source_any;
    bool final_card;

    PendingEvent();
};

struct LineMetrics {
    std::uint16_t district_mask;
    std::array<std::uint8_t, kDistrictCount> station_counts;
    int max_stations;
    int route;
    int thames_crossings;
    int tourist_visits;

    LineMetrics();
};

struct LineState {
    int start;
    std::uint64_t station_mask;
    Mask155 edge_mask;
    std::uint64_t leaf_mask;

    LineState();
};

struct ScoreDelta {
    int route;
    int thames;
    int tourist;
    int interchange;
    int objective;

    ScoreDelta();
    int total() const;
};

struct LineScore {
    int districts;
    int max_stations;
    int thames_crossings;
    int route;
    int thames;
    int total;

    LineScore();
};

struct FinalScore {
    int line_total;
    int tourist_visits;
    int tourist_bonus;
    int two_line_stations;
    int three_line_stations;
    int four_line_stations;
    int interchange_bonus;
    int objectives_completed;
    int objective_bonus;
    int total;

    FinalScore();
};

struct GameOptions {
    std::uint64_t seed;
    bool has_order;
    std::array<std::uint8_t, kColorCount> order;
    bool shared_objectives_enabled;
    bool pencil_powers_enabled;
    std::uint8_t objective_count;
    std::array<std::uint8_t, 2> objective_cards;
    bool has_power_assignments;
    std::array<std::int8_t, kColorCount> power_assignments;

    GameOptions();
};

/* Canonical pending-free public state used by replay and cross-language checks. */
struct PublicState {
    std::array<std::uint64_t, kColorCount> line_station_masks;
    std::array<Mask155, kColorCount> line_edge_masks;
    std::uint16_t remaining_mask;
    std::array<std::uint8_t, kColorCount> order;
    std::uint8_t round_index;
    std::uint8_t underground_count;
    std::uint8_t draw_count;
    bool terminated;

    PublicState();
};

struct PublicDraw {
    double probability;
    std::array<std::uint8_t, 2> card_ids;
    std::uint8_t count;
};

struct PublicDrawList {
    std::array<PublicDraw, kMaxPublicDrawCount> items;
    std::size_t count;
};

struct Candidate;
struct ChanceOutcome;

class GameState {
public:
    explicit GameState(std::uint64_t seed = 0);
    GameState(const std::array<std::uint8_t, kColorCount>& order,
              std::uint64_t seed = 0);
    explicit GameState(const GameOptions& options);

    void reset();
    void draw();
    void draw_known_cards(const std::vector<int>& card_ids);
    void draw_known_cards(int first_card_id, int second_card_id = -1);
    void restore_pending(const PendingEvent& event);

    Status status() const;
    bool terminated() const;
    bool has_pending() const;
    const PendingEvent& pending() const;
    std::uint16_t remaining_mask() const;
    std::uint8_t round_index() const;
    std::uint8_t underground_count() const;
    std::uint8_t draw_count() const;
    const std::array<std::uint8_t, kColorCount>& order() const;
    const std::array<LineState, kColorCount>& lines() const;
    const std::array<LineMetrics, kColorCount>& line_metrics() const;
    const std::array<std::uint8_t, kColorCount + 1>& interchange_counts() const;
    bool shared_objectives_enabled() const;
    bool pencil_powers_enabled() const;
    const std::array<std::uint8_t, 2>& objective_cards() const;
    std::uint8_t shared_objective_mask() const;
    const std::array<std::int8_t, kColorCount>& power_assignments() const;
    std::uint8_t used_power_mask() const;
    std::uint8_t completed_objective_mask() const;
    bool double_section_pending() const;
    Symbol double_target_symbol() const;
    const std::vector<LineScore>& round_scores() const;
    const std::vector<std::uint8_t>& hidden_deck_order() const;
    bool is_public_copy() const;
    std::string random_state() const;

    std::vector<Action> legal_actions() const;
    void legal_actions(std::vector<Action>* destination) const;
    std::vector<Action> legal_actions_for_event(
        Symbol target_symbol,
        bool wild,
        bool source_any) const;
    ScoreDelta score_delta(const Action& action) const;
    ScoreDelta score_delta_for_legal_action(const Action& action) const;
    void apply_action(const Action& action);
    void apply_action_unchecked(const Action& action);

    std::vector<Candidate> candidates() const;
    PublicDrawList public_draws() const;
    GameState public_successor(const PublicDraw& draw) const;
    std::vector<ChanceOutcome> public_successors() const;

    PublicState canonical() const;
    static GameState from_canonical(const PublicState& state);
    GameState copy_public() const;
    void configure_advanced(
        bool objectives_enabled,
        const std::array<std::uint8_t, 2>& objectives,
        bool powers_enabled,
        const std::array<std::int8_t, kColorCount>& powers);
    void restore_advanced_state(
        std::uint8_t used_power_mask,
        std::uint8_t completed_objective_mask,
        bool double_section_pending,
        Symbol double_target_symbol);
    void restore_hidden_state(
        const std::string& random_state,
        const std::vector<std::uint8_t>& deck_order,
        bool public_copy);
    std::string canonical_signature() const;

    std::array<int, 6> partial_score_components() const;
    FinalScore final_score() const;
    int current_total() const;
    void write_features(float* destination, std::size_t capacity) const;

private:
    struct HiddenState {
        std::mt19937_64 rng;
        std::vector<std::uint8_t> deck_order;

        explicit HiddenState(std::uint64_t seed) : rng(seed), deck_order() {}
    };

    const Map* map_;
    std::array<std::uint8_t, kColorCount> order_;
    std::array<LineState, kColorCount> lines_;
    std::array<LineMetrics, kColorCount> metrics_;
    Mask155 board_edges_;
    std::uint64_t network_station_mask_;
    std::uint16_t network_district_mask_;
    std::array<std::uint8_t, kStationCount> lines_per_station_;
    std::array<std::uint8_t, kColorCount + 1> interchange_counts_;
    int route_total_;
    int thames_total_;
    int partial_tourist_visits_;
    int partial_tourist_points_;
    int interchange_total_;
    int interchange_station_total_;
    std::vector<LineScore> round_scores_;
    bool shared_objectives_enabled_;
    bool pencil_powers_enabled_;
    std::array<std::uint8_t, 2> objective_cards_;
    std::uint8_t shared_objective_mask_;
    std::array<std::int8_t, kColorCount> power_assignments_;
    std::uint8_t used_power_mask_;
    std::uint8_t completed_objective_mask_;
    bool double_section_pending_;
    Symbol double_target_symbol_;
    std::uint16_t remaining_mask_;
    std::uint8_t round_index_;
    std::uint8_t underground_count_;
    std::uint8_t draw_count_;
    Status status_;
    bool has_pending_;
    PendingEvent pending_;
    std::shared_ptr<HiddenState> hidden_;

    void validate_order() const;
    void validate_advanced_config() const;
    void ensure_unique_hidden();
    void start_round();
    int draw_one_random();
    int draw_one_known(int card_id);
    void draw_known_card_ids(const int* card_ids, std::size_t card_count);
    void set_pending_from_cards(const int* card_ids, std::size_t card_count);
    void update_score_caches(const Action& action);
    void complete_turn();
    void finish_round();
    void rebuild_derived_state(bool terminated);
    void rebuild_round_scores(bool terminated);
    PencilPower active_power() const;
    bool power_available(PencilPower power) const;
    void mark_power_used(PencilPower power);
    std::uint8_t achieved_objective_mask(
        std::uint64_t network_station_mask,
        std::uint16_t network_district_mask,
        int interchange_stations,
        int thames_crossings) const;
    int circle_route_bonus() const;
    void append_section_actions(
        Symbol target_symbol,
        bool wild,
        bool source_any,
        PencilPower power,
        std::vector<Action>* destination) const;
    bool action_is_legal(const Action& action) const;
    ScoreDelta score_delta_unchecked(const Action& action) const;
};

struct Candidate {
    int action_index;
    Action action;
    int reward;
    GameState afterstate;

    Candidate(int index, const Action& selected, int value,
              const GameState& child);
};

struct ChanceOutcome {
    double probability;
    GameState state;

    ChanceOutcome(double probability_value, const GameState& child);
    ChanceOutcome(double probability_value, GameState&& child);
};

}  // namespace native
}  // namespace next_station

#endif  // NEXT_STATION_NATIVE_ENGINE_HPP
