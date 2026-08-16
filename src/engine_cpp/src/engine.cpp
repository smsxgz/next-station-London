#include "next_station/engine.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace next_station {
namespace native {

namespace {

const int kInterchangeTrack[kColorCount + 1] = {0, 0, 2, 5, 9};
const int kTouristTrack[11] = {0, 1, 2, 4, 6, 8, 11, 14, 17, 21, 25};
const std::uint16_t kFullCardMask = static_cast<std::uint16_t>((1u << kCardCount) - 1u);
const std::uint64_t kObjectiveSeedSalt = UINT64_C(0x9E3779B97F4A7C15);
const std::uint64_t kPowerSeedSalt = UINT64_C(0xD1B54A32D192ED03);

struct RawStation {
    int x;
    int y;
    Symbol symbol;
    bool tourist;
    int departure_color;
};

const RawStation kRawStations[kStationCount] = {
    {0, 0, kPentagon, false, -1},
    {1, 0, kTriangle, false, -1},
    {2, 0, kSquare, false, -1},
    {4, 0, kTriangle, false, -1},
    {5, 0, kCircle, false, -1},
    {7, 0, kTriangle, false, -1},
    {9, 0, kCircle, false, -1},
    {1, 1, kPentagon, false, -1},
    {3, 1, kSquare, false, -1},
    {6, 1, kPentagon, true, -1},
    {8, 1, kSquare, false, -1},
    {9, 1, kPentagon, false, -1},
    {0, 2, kCircle, false, -1},
    {3, 2, kTriangle, false, kGreen},
    {6, 2, kSquare, false, -1},
    {9, 2, kTriangle, false, -1},
    {0, 3, kSquare, true, -1},
    {2, 3, kPentagon, false, -1},
    {4, 3, kTriangle, false, -1},
    {5, 3, kCentral, true, -1},
    {6, 3, kCircle, false, -1},
    {7, 3, kCircle, false, kPink},
    {9, 3, kSquare, false, -1},
    {1, 4, kTriangle, false, -1},
    {2, 4, kSquare, false, -1},
    {4, 4, kPentagon, false, -1},
    {5, 4, kSquare, false, -1},
    {8, 4, kPentagon, false, -1},
    {0, 5, kPentagon, false, -1},
    {2, 5, kSquare, false, kPurple},
    {4, 5, kCircle, false, -1},
    {7, 5, kCircle, false, -1},
    {3, 6, kPentagon, false, -1},
    {4, 6, kTriangle, false, -1},
    {6, 6, kSquare, false, -1},
    {7, 6, kTriangle, false, -1},
    {9, 6, kTriangle, true, -1},
    {0, 7, kCircle, false, -1},
    {2, 7, kSquare, false, -1},
    {3, 7, kCircle, false, -1},
    {5, 7, kPentagon, false, kBlue},
    {8, 7, kCircle, false, -1},
    {9, 7, kPentagon, false, -1},
    {1, 8, kCircle, false, -1},
    {6, 8, kPentagon, false, -1},
    {8, 8, kTriangle, false, -1},
    {0, 9, kTriangle, false, -1},
    {1, 9, kSquare, false, -1},
    {3, 9, kPentagon, false, -1},
    {4, 9, kCircle, true, -1},
    {5, 9, kTriangle, false, -1},
    {7, 9, kCircle, false, -1},
    {9, 9, kSquare, false, -1},
};

const Card kDeck[kCardCount] = {
    {0, kCircle, true, false},
    {1, kTriangle, true, false},
    {2, kSquare, true, false},
    {3, kPentagon, true, false},
    {4, kWild, true, false},
    {5, kCircle, false, false},
    {6, kTriangle, false, false},
    {7, kSquare, false, false},
    {8, kPentagon, false, false},
    {9, kWild, false, false},
    {10, kWild, false, true},
};

std::string district_name(int x, int y) {
    if (x == 0 && y == 0) return "northwest";
    if (x == 9 && y == 0) return "northeast";
    if (x == 0 && y == 9) return "southwest";
    if (x == 9 && y == 9) return "southeast";
    const char* column = x <= 2 ? "west" : (x <= 6 ? "central" : "east");
    const char* row = y <= 2 ? "north" : (y <= 6 ? "middle" : "south");
    return std::string(row) + "_" + column;
}

std::string district_at_point(double x, double y) {
    if (x < 0.5 && y < 0.5) return "northwest";
    if (x > 8.5 && y < 0.5) return "northeast";
    if (x < 0.5 && y > 8.5) return "southwest";
    if (x > 8.5 && y > 8.5) return "southeast";
    const char* column = x < 2.5 ? "west" : (x < 6.5 ? "central" : "east");
    const char* row = y < 2.5 ? "north" : (y < 6.5 ? "middle" : "south");
    return std::string(row) + "_" + column;
}

double orientation(double ax, double ay, double bx, double by,
                    double cx, double cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

bool on_segment(double ax, double ay, double bx, double by,
                double px, double py) {
    const double eps = 1e-9;
    return std::fabs(orientation(ax, ay, bx, by, px, py)) <= eps
        && px >= std::min(ax, bx) - eps && px <= std::max(ax, bx) + eps
        && py >= std::min(ay, by) - eps && py <= std::max(ay, by) + eps;
}

bool segments_intersect(double ax, double ay, double bx, double by,
                        double cx, double cy, double dx, double dy) {
    const double eps = 1e-9;
    const double o1 = orientation(ax, ay, bx, by, cx, cy);
    const double o2 = orientation(ax, ay, bx, by, dx, dy);
    const double o3 = orientation(cx, cy, dx, dy, ax, ay);
    const double o4 = orientation(cx, cy, dx, dy, bx, by);
    if (((o1 > eps) != (o2 > eps)) && ((o3 > eps) != (o4 > eps))) {
        return true;
    }
    return (std::fabs(o1) <= eps && on_segment(ax, ay, bx, by, cx, cy))
        || (std::fabs(o2) <= eps && on_segment(ax, ay, bx, by, dx, dy))
        || (std::fabs(o3) <= eps && on_segment(cx, cy, dx, dy, ax, ay))
        || (std::fabs(o4) <= eps && on_segment(cx, cy, dx, dy, bx, by));
}

bool crosses_thames(const RawStation& first, const RawStation& second) {
    const double river[][4] = {
        {-1.0, 3.35, 2.05, 3.35},
        {2.05, 3.35, 3.65, 5.25},
        {3.65, 5.25, 5.05, 5.25},
        {5.05, 5.25, 6.45, 4.25},
        {6.45, 4.25, 10.0, 4.25},
    };
    for (std::size_t i = 0; i < sizeof(river) / sizeof(river[0]); ++i) {
        if (segments_intersect(
                first.x, first.y, second.x, second.y,
                river[i][0], river[i][1], river[i][2], river[i][3])) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> edge_district_names(const RawStation& first,
                                             const RawStation& second) {
    const double dx = static_cast<double>(second.x - first.x);
    const double dy = static_cast<double>(second.y - first.y);
    std::vector<double> parameters;
    parameters.push_back(0.0);
    parameters.push_back(1.0);
    const double boundaries[] = {0.5, 2.5, 6.5, 8.5};
    for (std::size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); ++i) {
        if (dx != 0.0) {
            const double t = (boundaries[i] - first.x) / dx;
            if (t > 0.0 && t < 1.0) parameters.push_back(t);
        }
        if (dy != 0.0) {
            const double t = (boundaries[i] - first.y) / dy;
            if (t > 0.0 && t < 1.0) parameters.push_back(t);
        }
    }
    std::sort(parameters.begin(), parameters.end());
    std::vector<double> unique_parameters;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (unique_parameters.empty()
            || std::fabs(unique_parameters.back() - parameters[i]) > 1e-12) {
            unique_parameters.push_back(parameters[i]);
        }
    }
    std::set<std::string> districts;
    districts.insert(district_name(first.x, first.y));
    districts.insert(district_name(second.x, second.y));
    for (std::size_t i = 0; i + 1 < unique_parameters.size(); ++i) {
        const double middle = (unique_parameters[i] + unique_parameters[i + 1]) / 2.0;
        districts.insert(district_at_point(
            first.x + dx * middle, first.y + dy * middle));
    }
    return std::vector<std::string>(districts.begin(), districts.end());
}

int coordinate_key(int x, int y) {
    return x * 16 + y;
}

struct TemporaryEdge {
    int u;
    int v;
    bool crosses;
    std::vector<std::string> districts;
};

bool edge_conflicts(const TemporaryEdge& first, const TemporaryEdge& second) {
    if (first.u == second.u || first.u == second.v
        || first.v == second.u || first.v == second.v) {
        return false;
    }
    const RawStation& a = kRawStations[first.u];
    const RawStation& b = kRawStations[first.v];
    const RawStation& c = kRawStations[second.u];
    const RawStation& d = kRawStations[second.v];
    return segments_intersect(
        a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
}

Map build_map() {
    Map result;
    result.stations.reserve(kStationCount);
    result.adjacency.resize(kStationCount);
    result.oriented_adjacency.resize(kStationCount);

    std::vector<std::string> station_district_names;
    station_district_names.reserve(kStationCount);
    std::map<int, int> coordinates;
    for (int id = 0; id < kStationCount; ++id) {
        const RawStation& raw = kRawStations[id];
        station_district_names.push_back(district_name(raw.x, raw.y));
        coordinates[coordinate_key(raw.x, raw.y)] = id;
        Station station;
        station.id = id;
        station.x = raw.x;
        station.y = raw.y;
        station.symbol = raw.symbol;
        station.district = -1;
        station.tourist = raw.tourist;
        station.departure_color = raw.departure_color;
        result.stations.push_back(station);
    }

    std::vector<TemporaryEdge> temporary_edges;
    for (int first_id = 0; first_id < kStationCount; ++first_id) {
        const RawStation& first = kRawStations[first_id];
        for (int second_id = first_id + 1; second_id < kStationCount; ++second_id) {
            const RawStation& second = kRawStations[second_id];
            const int dx = second.x - first.x;
            const int dy = second.y - first.y;
            if (!(dx == 0 || dy == 0 || std::abs(dx) == std::abs(dy))) continue;
            const int sx = (dx > 0) - (dx < 0);
            const int sy = (dy > 0) - (dy < 0);
            int x = first.x + sx;
            int y = first.y + sy;
            bool blocked = false;
            while (x != second.x || y != second.y) {
                if (coordinates.find(coordinate_key(x, y)) != coordinates.end()) {
                    blocked = true;
                    break;
                }
                x += sx;
                y += sy;
            }
            if (!blocked) {
                TemporaryEdge edge;
                edge.u = first_id;
                edge.v = second_id;
                edge.crosses = crosses_thames(first, second);
                edge.districts = edge_district_names(first, second);
                temporary_edges.push_back(edge);
            }
        }
    }

    std::set<std::string> district_set;
    for (std::size_t i = 0; i < station_district_names.size(); ++i) {
        district_set.insert(station_district_names[i]);
    }
    for (std::size_t i = 0; i < temporary_edges.size(); ++i) {
        district_set.insert(temporary_edges[i].districts.begin(),
                            temporary_edges[i].districts.end());
    }
    std::vector<std::string> district_names(district_set.begin(), district_set.end());
    std::map<std::string, int> district_indices;
    for (std::size_t i = 0; i < district_names.size(); ++i) {
        district_indices[district_names[i]] = static_cast<int>(i);
    }
    result.district_count = static_cast<int>(district_names.size());
    if (result.district_count != kDistrictCount) {
        throw std::runtime_error("London map district count differs from Python engine");
    }
    for (int id = 0; id < kStationCount; ++id) {
        const int district = district_indices[station_district_names[id]];
        result.stations[id].district = district;
        result.station_district_indices[id] = district;
    }

    result.edges.reserve(temporary_edges.size());
    result.conflict_masks.resize(temporary_edges.size());
    for (std::size_t id = 0; id < temporary_edges.size(); ++id) {
        const TemporaryEdge& source = temporary_edges[id];
        std::uint16_t mask = 0;
        for (std::size_t j = 0; j < source.districts.size(); ++j) {
            mask = static_cast<std::uint16_t>(
                mask | (1u << district_indices[source.districts[j]]));
        }
        mask = static_cast<std::uint16_t>(
            mask | (1u << result.stations[source.u].district)
            | (1u << result.stations[source.v].district));
        Edge edge;
        edge.id = static_cast<int>(id);
        edge.u = source.u;
        edge.v = source.v;
        edge.crosses_thames = source.crosses;
        edge.district_mask = mask;
        result.edges.push_back(edge);
        result.edge_district_masks[id] = mask;
    }
    if (result.edges.size() != kEdgeCount) {
        throw std::runtime_error("London map edge count differs from Python engine");
    }
    for (std::size_t i = 0; i < result.edges.size(); ++i) {
        const Edge& edge = result.edges[i];
        result.adjacency[edge.u].push_back(edge.id);
        result.adjacency[edge.v].push_back(edge.id);
        result.oriented_adjacency[edge.u].push_back(std::make_pair(edge.id, edge.v));
        result.oriented_adjacency[edge.v].push_back(std::make_pair(edge.id, edge.u));
        for (std::size_t j = 0; j < i; ++j) {
            if (edge_conflicts(temporary_edges[i], temporary_edges[j])) {
                result.conflict_masks[i].set(static_cast<int>(j));
                result.conflict_masks[j].set(static_cast<int>(i));
            }
        }
    }
    return result;
}

const Map& map_instance() {
    static const Map value = build_map();
    return value;
}

int popcount_u64(std::uint64_t value) {
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

Mask155::Mask155() : words{0, 0, 0} {}

Mask155::Mask155(std::uint64_t first_word) : words{first_word, 0, 0} {}

void Mask155::clear() {
    words[0] = words[1] = words[2] = 0;
}

bool Mask155::test(int index) const {
    require(index >= 0 && index < kEdgeCount, "edge bit index outside map");
    return (words[index / 64] & (std::uint64_t(1) << (index % 64))) != 0;
}

void Mask155::set(int index) {
    require(index >= 0 && index < kEdgeCount, "edge bit index outside map");
    words[index / 64] |= std::uint64_t(1) << (index % 64);
}

void Mask155::reset(int index) {
    require(index >= 0 && index < kEdgeCount, "edge bit index outside map");
    words[index / 64] &= ~(std::uint64_t(1) << (index % 64));
}

bool Mask155::intersects(const Mask155& other) const {
    return (words[0] & other.words[0]) != 0
        || (words[1] & other.words[1]) != 0
        || (words[2] & other.words[2]) != 0;
}

int Mask155::count() const {
    return popcount_u64(words[0]) + popcount_u64(words[1])
        + popcount_u64(words[2] & ((std::uint64_t(1) << 27) - 1));
}

Mask155& Mask155::operator|=(const Mask155& other) {
    words[0] |= other.words[0];
    words[1] |= other.words[1];
    words[2] |= other.words[2];
    return *this;
}

Mask155& Mask155::operator&=(const Mask155& other) {
    words[0] &= other.words[0];
    words[1] &= other.words[1];
    words[2] &= other.words[2];
    return *this;
}

bool operator==(const Mask155& left, const Mask155& right) {
    return left.words[0] == right.words[0]
        && left.words[1] == right.words[1]
        && (left.words[2] & ((std::uint64_t(1) << 27) - 1))
            == (right.words[2] & ((std::uint64_t(1) << 27) - 1));
}

bool operator!=(const Mask155& left, const Mask155& right) {
    return !(left == right);
}

Mask155 operator|(Mask155 left, const Mask155& right) {
    left |= right;
    return left;
}

Mask155 operator&(Mask155 left, const Mask155& right) {
    left &= right;
    return left;
}

const Map& london_map() {
    return map_instance();
}

const std::array<Card, kCardCount>& deck() {
    static const std::array<Card, kCardCount> value = {{
        {0, kCircle, true, false},
        {1, kTriangle, true, false},
        {2, kSquare, true, false},
        {3, kPentagon, true, false},
        {4, kWild, true, false},
        {5, kCircle, false, false},
        {6, kTriangle, false, false},
        {7, kSquare, false, false},
        {8, kPentagon, false, false},
        {9, kWild, false, false},
        {10, kWild, false, true},
    }};
    return value;
}

Action::Action() : edge_id(-1), source(-1), target(-1), power(kNoPower) {}

Action::Action(int edge, int from, int to, PencilPower selected_power)
    : edge_id(edge), source(from), target(to), power(selected_power) {}

bool Action::is_pass() const {
    return edge_id < 0;
}

bool operator==(const Action& left, const Action& right) {
    return left.edge_id == right.edge_id && left.source == right.source
        && left.target == right.target && left.power == right.power;
}

PendingEvent::PendingEvent()
    : card_ids{{0, 0}}, count(0), target_symbol(kWild), wild(false),
      source_any(false), final_card(false) {}

LineMetrics::LineMetrics()
    : district_mask(0), station_counts{{0}}, max_stations(0), route(0),
      thames_crossings(0), tourist_visits(0) {}

LineState::LineState()
    : start(-1), station_mask(0), edge_mask(), leaf_mask(0) {}

ScoreDelta::ScoreDelta()
    : route(0), thames(0), tourist(0), interchange(0), objective(0) {}

int ScoreDelta::total() const {
    return route + thames + tourist + interchange + objective;
}

LineScore::LineScore()
    : districts(0), max_stations(0), thames_crossings(0), route(0), thames(0),
      total(0) {}

FinalScore::FinalScore()
    : line_total(0), tourist_visits(0), tourist_bonus(0), two_line_stations(0),
      three_line_stations(0), four_line_stations(0), interchange_bonus(0),
      objectives_completed(0), objective_bonus(0), total(0) {}

GameOptions::GameOptions()
    : seed(0), has_order(false), order{{0, 1, 2, 3}},
      shared_objectives_enabled(false), pencil_powers_enabled(false),
      objective_count(0), objective_cards{{0, 0}},
      has_power_assignments(false),
      power_assignments{{kNoPower, kNoPower, kNoPower, kNoPower}} {}

PublicState::PublicState()
    : line_station_masks{{0, 0, 0, 0}}, line_edge_masks(), remaining_mask(0),
      order{{0, 1, 2, 3}}, round_index(0), underground_count(0),
      draw_count(0), terminated(false) {}

namespace {

GameOptions random_base_options(std::uint64_t seed) {
    GameOptions result;
    result.seed = seed;
    return result;
}

GameOptions ordered_base_options(
    const std::array<std::uint8_t, kColorCount>& order,
    std::uint64_t seed) {
    GameOptions result;
    result.seed = seed;
    result.has_order = true;
    result.order = order;
    return result;
}

}  // namespace

GameState::GameState(std::uint64_t seed)
    : GameState(random_base_options(seed)) {}

GameState::GameState(
    const std::array<std::uint8_t, kColorCount>& order, std::uint64_t seed)
    : GameState(ordered_base_options(order, seed)) {}

GameState::GameState(const GameOptions& options)
    : map_(&london_map()), order_{{0, 1, 2, 3}}, lines_(), metrics_(),
      board_edges_(), network_station_mask_(0), network_district_mask_(0),
      lines_per_station_{{0}}, interchange_counts_{{0}}, route_total_(0),
      thames_total_(0), partial_tourist_visits_(0), partial_tourist_points_(0),
      interchange_total_(0), interchange_station_total_(0), round_scores_(),
      shared_objectives_enabled_(options.shared_objectives_enabled),
      pencil_powers_enabled_(options.pencil_powers_enabled),
      objective_cards_{{0, 0}}, shared_objective_mask_(0),
      power_assignments_{{kNoPower, kNoPower, kNoPower, kNoPower}},
      used_power_mask_(0), completed_objective_mask_(0),
      double_section_pending_(false), double_target_symbol_(kWild),
      remaining_mask_(0), round_index_(0), underground_count_(0),
      draw_count_(0), status_(Status::Playing), has_pending_(false),
      pending_(), rng_(options.seed), deck_order_(), public_copy_(false) {
    if (options.has_order) {
        order_ = options.order;
    } else {
        for (int i = kColorCount - 1; i > 0; --i) {
            std::uniform_int_distribution<int> distribution(0, i);
            const int other = distribution(rng_);
            std::swap(order_[i], order_[other]);
        }
    }
    validate_order();

    if (shared_objectives_enabled_) {
        if (options.objective_count == 2) {
            objective_cards_ = options.objective_cards;
        } else {
            require(options.objective_count == 0,
                    "objective count must be zero or two");
            std::array<std::uint8_t, kObjectiveCount> objectives = {{0, 1, 2, 3, 4}};
            std::mt19937_64 objective_rng(options.seed ^ kObjectiveSeedSalt);
            std::shuffle(objectives.begin(), objectives.end(), objective_rng);
            objective_cards_[0] = objectives[0];
            objective_cards_[1] = objectives[1];
        }
        shared_objective_mask_ = static_cast<std::uint8_t>(
            (1u << objective_cards_[0]) | (1u << objective_cards_[1]));
    } else {
        require(options.objective_count == 0,
                "objective cards require shared objectives");
    }

    if (pencil_powers_enabled_) {
        if (options.has_power_assignments) {
            power_assignments_ = options.power_assignments;
        } else {
            std::array<std::int8_t, kColorCount> powers = {{
                kDoubleSection, kWildCard, kRailroadSwitch, kCircleStation,
            }};
            std::mt19937_64 power_rng(options.seed ^ kPowerSeedSalt);
            std::shuffle(powers.begin(), powers.end(), power_rng);
            power_assignments_ = powers;
        }
    } else {
        require(!options.has_power_assignments,
                "power assignments require pencil powers");
    }
    validate_advanced_config();
    reset();
}

void GameState::validate_order() const {
    bool seen[kColorCount] = {false, false, false, false};
    for (int i = 0; i < kColorCount; ++i) {
        require(order_[i] < kColorCount, "color order contains an invalid color");
        require(!seen[order_[i]], "color order contains a duplicate color");
        seen[order_[i]] = true;
    }
}

void GameState::validate_advanced_config() const {
    if (shared_objectives_enabled_) {
        require(objective_cards_[0] < kObjectiveCount
                    && objective_cards_[1] < kObjectiveCount,
                "objective card index is outside the deck");
        require(objective_cards_[0] != objective_cards_[1],
                "objective cards must be distinct");
    } else {
        require(shared_objective_mask_ == 0,
                "disabled objectives have a nonzero mask");
    }
    bool seen[kPowerCount] = {false, false, false, false};
    for (int color = 0; color < kColorCount; ++color) {
        const int power = power_assignments_[color];
        if (!pencil_powers_enabled_) {
            require(power == kNoPower,
                    "disabled pencil powers have an assignment");
            continue;
        }
        require(power >= 0 && power < kPowerCount,
                "pencil power index is outside the deck");
        require(!seen[power], "pencil power assignment contains a duplicate");
        seen[power] = true;
    }
}

void GameState::reset() {
    validate_order();
    validate_advanced_config();
    round_index_ = 0;
    used_power_mask_ = 0;
    completed_objective_mask_ = 0;
    double_section_pending_ = false;
    double_target_symbol_ = kWild;
    underground_count_ = 0;
    draw_count_ = 0;
    status_ = Status::Playing;
    has_pending_ = false;
    pending_ = PendingEvent();
    round_scores_.clear();
    board_edges_.clear();
    network_station_mask_ = 0;
    network_district_mask_ = 0;
    lines_per_station_.fill(0);
    interchange_counts_.fill(0);
    route_total_ = 0;
    thames_total_ = 0;
    partial_tourist_visits_ = 0;
    partial_tourist_points_ = 0;
    interchange_total_ = 0;
    interchange_station_total_ = 0;
    for (int color = 0; color < kColorCount; ++color) {
        int start = -1;
        for (int station = 0; station < kStationCount; ++station) {
            if (map_->stations[station].departure_color == color) {
                require(start < 0, "map contains multiple departure stations");
                start = station;
            }
        }
        require(start >= 0, "map is missing a departure station");
        lines_[color] = LineState();
        lines_[color].start = start;
        lines_[color].station_mask = std::uint64_t(1) << start;
        lines_[color].leaf_mask = lines_[color].station_mask;
        metrics_[color] = LineMetrics();
        const int district = map_->station_district_indices[start];
        metrics_[color].district_mask = static_cast<std::uint16_t>(1u << district);
        metrics_[color].station_counts[district] = 1;
        metrics_[color].max_stations = 1;
        metrics_[color].route = 1;
        metrics_[color].tourist_visits = map_->stations[start].tourist ? 1 : 0;
        network_station_mask_ |= std::uint64_t(1) << start;
        network_district_mask_ = static_cast<std::uint16_t>(
            network_district_mask_ | (1u << district));
        route_total_ += 1;
        partial_tourist_visits_ += metrics_[color].tourist_visits;
    }
    for (int station = 0; station < kStationCount; ++station) {
        const int count = popcount_u64(
            ((lines_[0].station_mask >> station) & 1u)
            | (((lines_[1].station_mask >> station) & 1u) << 1)
            | (((lines_[2].station_mask >> station) & 1u) << 2)
            | (((lines_[3].station_mask >> station) & 1u) << 3));
        lines_per_station_[station] = static_cast<std::uint8_t>(count);
        ++interchange_counts_[count];
    }
    partial_tourist_points_ = kTouristTrack[
        std::min(partial_tourist_visits_, 10)];
    for (int count = 0; count <= kColorCount; ++count) {
        interchange_total_ += interchange_counts_[count] * kInterchangeTrack[count];
    }
    for (int count = 2; count <= kColorCount; ++count) {
        interchange_station_total_ += interchange_counts_[count];
    }
    completed_objective_mask_ = static_cast<std::uint8_t>(
        achieved_objective_mask(
            network_station_mask_, network_district_mask_,
            interchange_station_total_, 0)
        & shared_objective_mask_);
    public_copy_ = false;
    start_round();
}

void GameState::start_round() {
    remaining_mask_ = kFullCardMask;
    deck_order_.clear();
    for (int card = 0; card < kCardCount; ++card) {
        deck_order_.push_back(static_cast<std::uint8_t>(card));
    }
    if (!public_copy_) {
        std::shuffle(deck_order_.begin(), deck_order_.end(), rng_);
    }
}

int GameState::draw_one_random() {
    require(!public_copy_, "a public state cannot draw a hidden random card");
    while (!deck_order_.empty()
           && (remaining_mask_ & (std::uint16_t(1) << deck_order_.front())) == 0) {
        deck_order_.erase(deck_order_.begin());
    }
    int card_id = -1;
    if (!deck_order_.empty()) {
        card_id = deck_order_.front();
        deck_order_.erase(deck_order_.begin());
    } else {
        std::vector<int> choices;
        for (int id = 0; id < kCardCount; ++id) {
            if (remaining_mask_ & (std::uint16_t(1) << id)) choices.push_back(id);
        }
        require(!choices.empty(), "cannot draw from an empty card pile");
        std::uniform_int_distribution<int> distribution(
            0, static_cast<int>(choices.size()) - 1);
        card_id = choices[distribution(rng_)];
    }
    return draw_one_known(card_id);
}

int GameState::draw_one_known(int card_id) {
    require(card_id >= 0 && card_id < kCardCount, "invalid card id");
    const std::uint16_t bit = static_cast<std::uint16_t>(1u << card_id);
    require((remaining_mask_ & bit) != 0, "card is not in the remaining pile");
    remaining_mask_ = static_cast<std::uint16_t>(remaining_mask_ & ~bit);
    for (std::vector<std::uint8_t>::iterator it = deck_order_.begin();
         it != deck_order_.end(); ++it) {
        if (*it == card_id) {
            deck_order_.erase(it);
            break;
        }
    }
    ++draw_count_;
    if (deck()[card_id].underground) ++underground_count_;
    return card_id;
}

void GameState::set_pending_from_cards(const std::vector<int>& card_ids) {
    require(!card_ids.empty() && card_ids.size() <= 2,
            "a public event must contain one or two cards");
    const Card& first = deck()[card_ids[0]];
    const std::size_t expected = first.is_switch ? 2u : 1u;
    require(card_ids.size() == expected, "wrong number of cards for public event");
    pending_ = PendingEvent();
    pending_.count = static_cast<std::uint8_t>(card_ids.size());
    for (std::size_t i = 0; i < card_ids.size(); ++i) {
        pending_.card_ids[i] = static_cast<std::uint8_t>(card_ids[i]);
    }
    const int target_id = first.is_switch ? card_ids[1] : card_ids[0];
    pending_.target_symbol = deck()[target_id].symbol;
    pending_.wild = pending_.target_symbol == kWild;
    pending_.source_any = first.is_switch;
    pending_.final_card = underground_count_ >= 5;
    has_pending_ = true;
}

void GameState::draw() {
    require(status_ == Status::Playing && !has_pending_,
            "draw requires a playing state without a pending event");
    std::vector<int> cards;
    const int first = draw_one_random();
    cards.push_back(first);
    if (deck()[first].is_switch) {
        cards.push_back(draw_one_random());
    }
    set_pending_from_cards(cards);
}

void GameState::draw_known_cards(const std::vector<int>& card_ids) {
    require(status_ == Status::Playing && !has_pending_,
            "known draw requires a playing state without a pending event");
    require(!card_ids.empty(), "known draw must contain at least one card");
    bool seen[kCardCount] = {false, false, false, false, false, false,
                             false, false, false, false, false};
    for (std::size_t i = 0; i < card_ids.size(); ++i) {
        require(card_ids[i] >= 0 && card_ids[i] < kCardCount,
                "known draw contains an invalid card");
        require(!seen[card_ids[i]], "known draw repeats a card");
        seen[card_ids[i]] = true;
    }
    const std::size_t expected = deck()[card_ids[0]].is_switch ? 2u : 1u;
    require(card_ids.size() == expected, "known draw has the wrong card count");
    for (std::size_t i = 0; i < card_ids.size(); ++i) draw_one_known(card_ids[i]);
    set_pending_from_cards(card_ids);
}

void GameState::restore_pending(const PendingEvent& event) {
    require(status_ == Status::Playing && !has_pending_,
            "pending restoration requires a pending-free playing state");
    require(event.count == 1 || event.count == 2,
            "restored pending event has an invalid card count");
    for (int index = 0; index < event.count; ++index) {
        const int card_id = event.card_ids[index];
        require(card_id >= 0 && card_id < kCardCount,
                "restored pending event has an invalid card id");
        require((remaining_mask_ & (std::uint16_t(1) << card_id)) == 0,
                "restored pending card is still in the remaining pile");
    }
    require(event.count != 2 || event.card_ids[0] != event.card_ids[1],
            "restored pending event repeats a card");
    pending_ = event;
    has_pending_ = true;
}

Status GameState::status() const { return status_; }

bool GameState::terminated() const { return status_ == Status::Finished; }

bool GameState::has_pending() const { return has_pending_; }

const PendingEvent& GameState::pending() const {
    require(has_pending_, "state has no pending event");
    return pending_;
}

std::uint16_t GameState::remaining_mask() const { return remaining_mask_; }

std::uint8_t GameState::round_index() const { return round_index_; }

std::uint8_t GameState::underground_count() const { return underground_count_; }

std::uint8_t GameState::draw_count() const { return draw_count_; }

const std::array<std::uint8_t, kColorCount>& GameState::order() const {
    return order_;
}

const std::array<LineState, kColorCount>& GameState::lines() const {
    return lines_;
}

const std::array<LineMetrics, kColorCount>& GameState::line_metrics() const {
    return metrics_;
}

const std::array<std::uint8_t, kColorCount + 1>&
GameState::interchange_counts() const {
    return interchange_counts_;
}

bool GameState::shared_objectives_enabled() const {
    return shared_objectives_enabled_;
}

bool GameState::pencil_powers_enabled() const {
    return pencil_powers_enabled_;
}

const std::array<std::uint8_t, 2>& GameState::objective_cards() const {
    return objective_cards_;
}

std::uint8_t GameState::shared_objective_mask() const {
    return shared_objective_mask_;
}

const std::array<std::int8_t, kColorCount>&
GameState::power_assignments() const {
    return power_assignments_;
}

std::uint8_t GameState::used_power_mask() const {
    return used_power_mask_;
}

std::uint8_t GameState::completed_objective_mask() const {
    return completed_objective_mask_;
}

bool GameState::double_section_pending() const {
    return double_section_pending_;
}

Symbol GameState::double_target_symbol() const {
    return double_target_symbol_;
}

const std::vector<LineScore>& GameState::round_scores() const {
    return round_scores_;
}

const std::vector<std::uint8_t>& GameState::hidden_deck_order() const {
    return deck_order_;
}

bool GameState::is_public_copy() const {
    return public_copy_;
}

std::string GameState::random_state() const {
    std::ostringstream output;
    output << rng_;
    return output.str();
}

PencilPower GameState::active_power() const {
    if (!pencil_powers_enabled_) return kNoPower;
    return static_cast<PencilPower>(power_assignments_[order_[round_index_]]);
}

bool GameState::power_available(PencilPower power) const {
    return active_power() == power
        && (used_power_mask_ & (std::uint8_t(1u) << power)) == 0;
}

void GameState::mark_power_used(PencilPower power) {
    require(power >= 0 && power < kPowerCount,
            "cannot mark an invalid pencil power");
    require(power_available(power), "that pencil power is not available");
    used_power_mask_ = static_cast<std::uint8_t>(
        used_power_mask_ | (std::uint8_t(1u) << power));
}

std::uint8_t GameState::achieved_objective_mask(
    std::uint64_t network_station_mask,
    std::uint16_t network_district_mask,
    int interchange_stations,
    int thames_crossings) const {
    std::uint8_t result = 0;
    if (interchange_stations >= 8) result |= std::uint8_t(1u) << kEightInterchanges;
    const std::uint16_t all_districts = static_cast<std::uint16_t>(
        (1u << map_->district_count) - 1u);
    if (network_district_mask == all_districts) {
        result |= std::uint8_t(1u) << kAllDistricts;
    }
    std::uint64_t tourist_mask = 0;
    std::uint64_t central_mask = 0;
    const int middle_central = map_->station_district_indices[25];
    for (int station = 0; station < kStationCount; ++station) {
        if (map_->stations[station].tourist) {
            tourist_mask |= std::uint64_t(1) << station;
        }
        if (map_->station_district_indices[station] == middle_central) {
            central_mask |= std::uint64_t(1) << station;
        }
    }
    if ((network_station_mask & tourist_mask) == tourist_mask) {
        result |= std::uint8_t(1u) << kAllTouristSites;
    }
    if ((network_station_mask & central_mask) == central_mask) {
        result |= std::uint8_t(1u) << kAllCentralStations;
    }
    if (thames_crossings >= 6) {
        result |= std::uint8_t(1u) << kSixThamesCrossings;
    }
    return result;
}

int GameState::circle_route_bonus() const {
    if (!pencil_powers_enabled_) return 0;
    int result = 0;
    for (int color = 0; color < kColorCount; ++color) {
        if (power_assignments_[color] == kCircleStation) {
            result += popcount_u64(metrics_[color].district_mask);
        }
    }
    return result;
}

std::vector<Action> GameState::section_actions(
    Symbol target_symbol,
    bool wild,
    bool source_any,
    PencilPower power) const {
    const int color = order_[round_index_];
    const LineState& line = lines_[color];
    const std::uint64_t source_mask = source_any
        ? line.station_mask : line.leaf_mask;
    std::vector<Action> actions;
    for (int source = 0; source < kStationCount; ++source) {
        if ((source_mask & (std::uint64_t(1) << source)) == 0) continue;
        const std::vector<std::pair<int, int> >& adjacent =
            map_->oriented_adjacency[source];
        for (std::size_t i = 0; i < adjacent.size(); ++i) {
            const int edge_id = adjacent[i].first;
            const int target = adjacent[i].second;
            if (board_edges_.test(edge_id)
                || map_->conflict_masks[edge_id].intersects(board_edges_)) {
                continue;
            }
            if (line.station_mask & (std::uint64_t(1) << target)) continue;
            const Station& station = map_->stations[target];
            if (station.symbol != kCentral && !wild
                && station.symbol != target_symbol) {
                continue;
            }
            actions.push_back(Action(edge_id, source, target, power));
        }
    }
    return actions;
}

std::vector<Action> GameState::legal_actions() const {
    require(status_ == Status::Playing && has_pending_,
            "legal actions require a pending decision");
    if (double_section_pending_) {
        bool switch_revealed = false;
        for (int index = 0; index < pending_.count; ++index) {
            if (deck()[pending_.card_ids[index]].is_switch) switch_revealed = true;
        }
        return section_actions(
            double_target_symbol_, false,
            pending_.source_any || switch_revealed,
            kDoubleSection);
    }

    const std::vector<Action> base = section_actions(
        pending_.target_symbol, pending_.wild, pending_.source_any, kNoPower);
    const PencilPower power = active_power();
    if ((power != kWildCard && power != kRailroadSwitch)
        || !power_available(power)) {
        return base;
    }
    if (power == kRailroadSwitch && draw_count_ <= 2) return base;

    const std::vector<Action> powered = section_actions(
        power == kWildCard ? kWild : pending_.target_symbol,
        power == kWildCard || pending_.wild,
        power == kRailroadSwitch ? true : pending_.source_any,
        power);
    std::vector<Action> result = base;
    for (std::size_t index = 0; index < powered.size(); ++index) {
        bool duplicate_geometry = false;
        for (std::size_t base_index = 0; base_index < base.size(); ++base_index) {
            if (powered[index].edge_id == base[base_index].edge_id
                && powered[index].source == base[base_index].source
                && powered[index].target == base[base_index].target) {
                duplicate_geometry = true;
                break;
            }
        }
        if (!duplicate_geometry) result.push_back(powered[index]);
    }
    return result;
}

bool GameState::action_is_legal(const Action& action) const {
    if (action.is_pass()) return true;
    const std::vector<Action> actions = legal_actions();
    return std::find(actions.begin(), actions.end(), action) != actions.end();
}

ScoreDelta GameState::score_delta_unchecked(const Action& action) const {
    require(!action.is_pass(), "pass has no section score delta");
    const int color = order_[round_index_];
    const LineMetrics& metrics = metrics_[color];
    const Edge& edge = map_->edges[action.edge_id];
    const Station& target = map_->stations[action.target];
    const int target_district = map_->station_district_indices[action.target];
    const std::uint16_t districts_after = static_cast<std::uint16_t>(
        metrics.district_mask | map_->edge_district_masks[action.edge_id]);
    const int district_count = popcount_u64(districts_after);
    const int max_stations_after = std::max(
        metrics.max_stations,
        static_cast<int>(metrics.station_counts[target_district]) + 1);
    ScoreDelta result;
    result.route = district_count * max_stations_after - metrics.route;
    if (active_power() == kCircleStation) {
        result.route += district_count - popcount_u64(metrics.district_mask);
    }
    result.thames = edge.crosses_thames ? 2 : 0;
    if (target.tourist) {
        const int visits_after = partial_tourist_visits_ + 1;
        result.tourist = kTouristTrack[std::min(visits_after, 10)]
            - partial_tourist_points_;
    }
    const int before = lines_per_station_[action.target];
    const int after = before + 1;
    require(after <= kColorCount, "station is used by too many lines");
    result.interchange = kInterchangeTrack[after] - kInterchangeTrack[before];
    const std::uint64_t network_station_mask =
        network_station_mask_ | (std::uint64_t(1) << action.target);
    const std::uint16_t network_district_mask = static_cast<std::uint16_t>(
        network_district_mask_ | (std::uint16_t(1u) << target_district));
    const int interchange_stations = interchange_station_total_ + (before == 1 ? 1 : 0);
    const int thames_crossings = thames_total_ / 2 + (edge.crosses_thames ? 1 : 0);
    const std::uint8_t achieved = static_cast<std::uint8_t>(
        achieved_objective_mask(
            network_station_mask, network_district_mask,
            interchange_stations, thames_crossings)
        & shared_objective_mask_);
    result.objective = popcount_u64(
        static_cast<std::uint8_t>(achieved & ~completed_objective_mask_)) * 10;
    return result;
}

ScoreDelta GameState::score_delta(const Action& action) const {
    require(action_is_legal(action), "score delta requires a legal action");
    if (action.is_pass()) return ScoreDelta();
    return score_delta_unchecked(action);
}

void GameState::update_score_caches(const Action& action) {
    const int color = order_[round_index_];
    const ScoreDelta delta = score_delta_unchecked(action);
    LineState& line = lines_[color];
    LineMetrics& metrics = metrics_[color];
    const Edge& edge = map_->edges[action.edge_id];
    const Station& target = map_->stations[action.target];
    const int target_district = map_->station_district_indices[action.target];
    const std::uint16_t districts_after = static_cast<std::uint16_t>(
        metrics.district_mask | map_->edge_district_masks[action.edge_id]);
    const int max_stations_after = std::max(
        metrics.max_stations,
        static_cast<int>(metrics.station_counts[target_district]) + 1);

    metrics.district_mask = districts_after;
    ++metrics.station_counts[target_district];
    metrics.max_stations = max_stations_after;
    const int base_route_delta = popcount_u64(districts_after)
        * max_stations_after - metrics.route;
    metrics.route += base_route_delta;
    metrics.thames_crossings += edge.crosses_thames ? 1 : 0;
    if (target.tourist) ++metrics.tourist_visits;
    route_total_ += base_route_delta;
    thames_total_ += delta.thames;
    if (target.tourist) {
        ++partial_tourist_visits_;
        partial_tourist_points_ += delta.tourist;
    }

    const int before = lines_per_station_[action.target];
    require(before < kColorCount, "station is used by too many lines");
    if (before > 0) --interchange_counts_[before];
    const int after = before + 1;
    lines_per_station_[action.target] = static_cast<std::uint8_t>(after);
    ++interchange_counts_[after];
    interchange_total_ += delta.interchange;
    if (before == 1) ++interchange_station_total_;

    network_station_mask_ |= std::uint64_t(1) << action.target;
    network_district_mask_ = static_cast<std::uint16_t>(
        network_district_mask_ | (1u << target_district));
    completed_objective_mask_ = static_cast<std::uint8_t>(
        achieved_objective_mask(
            network_station_mask_, network_district_mask_,
            interchange_station_total_, thames_total_ / 2)
        & shared_objective_mask_);

    if (line.edge_mask.count() > 0) {
        line.leaf_mask &= ~(std::uint64_t(1) << action.source);
    }
    line.leaf_mask |= std::uint64_t(1) << action.target;
    line.edge_mask.set(action.edge_id);
    line.station_mask |= std::uint64_t(1) << action.target;
    board_edges_.set(action.edge_id);
}

void GameState::apply_action_unchecked(const Action& action) {
    require(status_ == Status::Playing && has_pending_,
            "cannot apply an action in the current state");
    if (double_section_pending_) {
        if (!action.is_pass()) {
            require(action.power == kDoubleSection,
                    "the optional second section must use its power");
            update_score_caches(action);
            mark_power_used(kDoubleSection);
        }
        complete_turn();
        return;
    }

    if (action.is_pass()) {
        complete_turn();
        return;
    }
    require(action.power != kDoubleSection,
            "a second-section action is not legal in the main phase");
    update_score_caches(action);
    if (action.power != kNoPower) mark_power_used(action.power);

    if (power_available(kDoubleSection)) {
        double_target_symbol_ = pending_.wild
            ? map_->stations[action.target].symbol
            : pending_.target_symbol;
        double_section_pending_ = true;
        if (!legal_actions().empty()) return;
        double_section_pending_ = false;
        double_target_symbol_ = kWild;
    }
    complete_turn();
}

void GameState::apply_action(const Action& action) {
    require(status_ == Status::Playing && has_pending_,
            "cannot apply an action in the current state");
    require(action_is_legal(action), "action is not legal in the current state");
    apply_action_unchecked(action);
}

void GameState::complete_turn() {
    const bool final_card = pending_.final_card;
    double_section_pending_ = false;
    double_target_symbol_ = kWild;
    has_pending_ = false;
    pending_ = PendingEvent();
    if (final_card) finish_round();
}

void GameState::finish_round() {
    const int color = order_[round_index_];
    const LineMetrics& metrics = metrics_[color];
    int circle_bonus = 0;
    LineScore score;
    score.districts = popcount_u64(metrics.district_mask);
    score.max_stations = metrics.max_stations;
    if (power_available(kCircleStation)) {
        circle_bonus = score.districts;
        ++score.max_stations;
        mark_power_used(kCircleStation);
    }
    score.thames_crossings = metrics.thames_crossings;
    score.route = metrics.route + circle_bonus;
    score.thames = metrics.thames_crossings * 2;
    score.total = score.route + score.thames;
    round_scores_.push_back(score);
    if (round_index_ == kColorCount - 1) {
        status_ = Status::Finished;
        return;
    }
    ++round_index_;
    underground_count_ = 0;
    draw_count_ = 0;
    start_round();
}

std::vector<Candidate> GameState::candidates() const {
    require(status_ == Status::Playing && has_pending_,
            "candidate generation requires a pending decision");
    const std::vector<Action> actions = legal_actions();
    std::vector<Candidate> result;
    result.reserve(actions.size() + 1);

    {
        GameState child = copy_public();
        child.apply_action_unchecked(Action());
        result.push_back(Candidate(kPassAction, Action(), 0, child));
    }
    for (std::size_t i = 0; i < actions.size(); ++i) {
        const Action& action = actions[i];
        const ScoreDelta delta = score_delta_unchecked(action);
        GameState child = copy_public();
        child.apply_action_unchecked(action);
        result.push_back(Candidate(
            action.edge_id, action, delta.total(), child));
    }
    std::sort(result.begin(), result.end(),
              [](const Candidate& left, const Candidate& right) {
                  if (left.action_index != right.action_index) {
                      return left.action_index < right.action_index;
                  }
                  if (left.action.source != right.action.source) {
                      return left.action.source < right.action.source;
                  }
                  return left.action.target < right.action.target;
              });
    return result;
}

std::vector<ChanceOutcome> GameState::public_successors() const {
    require(status_ == Status::Playing && !has_pending_,
            "chance expansion requires a pending-free playing state");
    std::vector<int> remaining;
    for (int card = 0; card < kCardCount; ++card) {
        if (remaining_mask_ & (std::uint16_t(1) << card)) remaining.push_back(card);
    }
    require(!remaining.empty(), "cannot expand an empty card pile");
    const double first_probability = 1.0 / static_cast<double>(remaining.size());
    std::vector<ChanceOutcome> result;
    for (std::size_t i = 0; i < remaining.size(); ++i) {
        const int first = remaining[i];
        if (!deck()[first].is_switch) {
            GameState child = copy_public();
            child.draw_known_cards(std::vector<int>(1, first));
            result.push_back(ChanceOutcome(first_probability, child));
            continue;
        }
        std::vector<int> following;
        for (std::size_t j = 0; j < remaining.size(); ++j) {
            if (j != i) following.push_back(remaining[j]);
        }
        require(!following.empty(), "switch card has no following card");
        const double probability = first_probability
            / static_cast<double>(following.size());
        for (std::size_t j = 0; j < following.size(); ++j) {
            GameState child = copy_public();
            std::vector<int> cards;
            cards.push_back(first);
            cards.push_back(following[j]);
            child.draw_known_cards(cards);
            result.push_back(ChanceOutcome(probability, child));
        }
    }
    return result;
}

PublicState GameState::canonical() const {
    require(!has_pending_, "canonical afterstate cannot contain a pending event");
    PublicState result;
    for (int color = 0; color < kColorCount; ++color) {
        result.line_station_masks[color] = lines_[color].station_mask;
        result.line_edge_masks[color] = lines_[color].edge_mask;
        result.order[color] = order_[color];
    }
    result.remaining_mask = remaining_mask_;
    result.round_index = round_index_;
    result.underground_count = underground_count_;
    result.draw_count = draw_count_;
    result.terminated = terminated();
    return result;
}

void GameState::rebuild_derived_state(bool terminated_value) {
    board_edges_.clear();
    network_station_mask_ = 0;
    network_district_mask_ = 0;
    route_total_ = 0;
    thames_total_ = 0;
    partial_tourist_visits_ = 0;
    partial_tourist_points_ = 0;
    interchange_total_ = 0;
    interchange_station_total_ = 0;
    lines_per_station_.fill(0);
    interchange_counts_.fill(0);

    for (int color = 0; color < kColorCount; ++color) {
        LineState& line = lines_[color];
        require(line.start >= 0 && line.start < kStationCount,
                "canonical line has an invalid departure station");
        require((line.station_mask & (std::uint64_t(1) << line.start)) != 0,
                "canonical line is missing its departure station");
        line.leaf_mask = 0;
        if (line.edge_mask.count() == 0) {
            line.leaf_mask = std::uint64_t(1) << line.start;
        } else {
            std::array<int, kStationCount> degree;
            degree.fill(0);
            for (int edge_id = 0; edge_id < kEdgeCount; ++edge_id) {
                if (!line.edge_mask.test(edge_id)) continue;
                const Edge& edge = map_->edges[edge_id];
                ++degree[edge.u];
                ++degree[edge.v];
                board_edges_.set(edge_id);
            }
            for (int station = 0; station < kStationCount; ++station) {
                if ((line.station_mask & (std::uint64_t(1) << station))
                    && degree[station] <= 1) {
                    line.leaf_mask |= std::uint64_t(1) << station;
                }
            }
        }
        LineMetrics& metrics = metrics_[color];
        metrics = LineMetrics();
        for (int station = 0; station < kStationCount; ++station) {
            if ((line.station_mask & (std::uint64_t(1) << station)) == 0) continue;
            const Station& value = map_->stations[station];
            const int district = map_->station_district_indices[station];
            metrics.district_mask = static_cast<std::uint16_t>(
                metrics.district_mask | (1u << district));
            ++metrics.station_counts[district];
            if (value.tourist) ++metrics.tourist_visits;
            network_station_mask_ |= std::uint64_t(1) << station;
            ++lines_per_station_[station];
        }
        for (int edge_id = 0; edge_id < kEdgeCount; ++edge_id) {
            if (!line.edge_mask.test(edge_id)) continue;
            const Edge& edge = map_->edges[edge_id];
            metrics.district_mask = static_cast<std::uint16_t>(
                metrics.district_mask | edge.district_mask);
            if (edge.crosses_thames) ++metrics.thames_crossings;
        }
        for (int district = 0; district < kDistrictCount; ++district) {
            metrics.max_stations = std::max(
                metrics.max_stations,
                static_cast<int>(metrics.station_counts[district]));
        }
        metrics.route = popcount_u64(metrics.district_mask)
            * metrics.max_stations;
        route_total_ += metrics.route;
        thames_total_ += metrics.thames_crossings * 2;
        partial_tourist_visits_ += metrics.tourist_visits;
        network_district_mask_ = static_cast<std::uint16_t>(
            network_district_mask_ | metrics.district_mask);
    }
    for (int station = 0; station < kStationCount; ++station) {
        ++interchange_counts_[lines_per_station_[station]];
    }
    partial_tourist_points_ = kTouristTrack[
        std::min(partial_tourist_visits_, 10)];
    for (int count = 0; count <= kColorCount; ++count) {
        interchange_total_ += interchange_counts_[count] * kInterchangeTrack[count];
        if (count >= 2) interchange_station_total_ += interchange_counts_[count];
    }

    completed_objective_mask_ = static_cast<std::uint8_t>(
        achieved_objective_mask(
            network_station_mask_, network_district_mask_,
            interchange_station_total_, thames_total_ / 2)
        & shared_objective_mask_);
    rebuild_round_scores(terminated_value);
    status_ = terminated_value ? Status::Finished : Status::Playing;
    has_pending_ = false;
    pending_ = PendingEvent();
    double_section_pending_ = false;
    double_target_symbol_ = kWild;
    deck_order_.clear();
    public_copy_ = true;
    /* The hidden random stream is intentionally absent from a public state. */
    rng_.seed(0);
}

void GameState::rebuild_round_scores(bool terminated_value) {
    round_scores_.clear();
    const int completed_count = terminated_value
        ? static_cast<int>(round_index_) + 1 : static_cast<int>(round_index_);
    require(completed_count >= 0 && completed_count <= kColorCount,
            "canonical round index is outside the game");
    int completed_tourists = 0;
    for (int i = 0; i < completed_count; ++i) {
        const int color = order_[i];
        const LineMetrics& metrics = metrics_[color];
        LineScore score;
        score.districts = popcount_u64(metrics.district_mask);
        score.max_stations = metrics.max_stations;
        const int circle_bonus =
            pencil_powers_enabled_
            && power_assignments_[color] == kCircleStation
            ? score.districts : 0;
        if (circle_bonus != 0) ++score.max_stations;
        score.thames_crossings = metrics.thames_crossings;
        score.route = metrics.route + circle_bonus;
        score.thames = metrics.thames_crossings * 2;
        score.total = score.route + score.thames;
        round_scores_.push_back(score);
        completed_tourists += metrics.tourist_visits;
    }
}

GameState GameState::from_canonical(const PublicState& state) {
    require(state.round_index < kColorCount,
            "canonical round index is outside the game");
    require(state.underground_count <= 5,
            "canonical underground count is outside the game");
    require(state.draw_count <= kCardCount,
            "canonical draw count is outside the game");
    require((state.remaining_mask & ~kFullCardMask) == 0,
            "canonical remaining mask is outside the deck");
    if (state.terminated) {
        require(state.round_index == kColorCount - 1,
                "only the final round can be terminal");
    }
    const std::uint64_t station_mask_limit =
        (std::uint64_t(1) << kStationCount) - 1;
    const std::uint64_t final_edge_word_limit =
        (std::uint64_t(1) << (kEdgeCount - 128)) - 1;
    for (int color = 0; color < kColorCount; ++color) {
        require((state.line_station_masks[color] & ~station_mask_limit) == 0,
                "canonical station mask is outside the map");
        require((state.line_edge_masks[color].words[2]
                 & ~final_edge_word_limit) == 0,
                "canonical edge mask is outside the map");
    }
    GameState result(state.order, 0);
    result.round_index_ = state.round_index;
    result.underground_count_ = state.underground_count;
    result.draw_count_ = state.draw_count;
    result.remaining_mask_ = state.remaining_mask;
    for (int color = 0; color < kColorCount; ++color) {
        result.lines_[color].station_mask = state.line_station_masks[color];
        result.lines_[color].edge_mask = state.line_edge_masks[color];
    }
    result.rebuild_derived_state(state.terminated);
    return result;
}

GameState GameState::copy_public() const {
    GameState result(*this);
    result.deck_order_.clear();
    result.public_copy_ = true;
    result.rng_.seed(0);
    return result;
}

void GameState::configure_advanced(
    bool objectives_enabled,
    const std::array<std::uint8_t, 2>& objectives,
    bool powers_enabled,
    const std::array<std::int8_t, kColorCount>& powers) {
    shared_objectives_enabled_ = objectives_enabled;
    objective_cards_ = objectives;
    shared_objective_mask_ = objectives_enabled
        ? static_cast<std::uint8_t>(
            (1u << objectives[0]) | (1u << objectives[1]))
        : 0;
    pencil_powers_enabled_ = powers_enabled;
    power_assignments_ = powers;
    if (!powers_enabled) power_assignments_.fill(kNoPower);
    validate_advanced_config();
    completed_objective_mask_ = static_cast<std::uint8_t>(
        achieved_objective_mask(
            network_station_mask_, network_district_mask_,
            interchange_station_total_, thames_total_ / 2)
        & shared_objective_mask_);
    rebuild_round_scores(terminated());
}

void GameState::restore_advanced_state(
    std::uint8_t used_power_mask,
    std::uint8_t completed_objective_mask,
    bool double_section_pending,
    Symbol double_target_symbol) {
    require((used_power_mask & ~((1u << kPowerCount) - 1u)) == 0,
            "used pencil-power mask is outside the power deck");
    require((completed_objective_mask & ~shared_objective_mask_) == 0,
            "completed objective mask is outside the selected cards");
    require(!double_section_pending || pencil_powers_enabled_,
            "double-section phase requires pencil powers");
    used_power_mask_ = used_power_mask;
    completed_objective_mask_ = completed_objective_mask;
    double_section_pending_ = double_section_pending;
    double_target_symbol_ = double_target_symbol;
}

void GameState::restore_hidden_state(
    const std::string& state,
    const std::vector<std::uint8_t>& deck_order,
    bool public_copy) {
    std::istringstream input(state);
    input >> rng_;
    require(!input.fail(), "serialized random state is invalid");
    bool seen[kCardCount] = {false, false, false, false, false, false,
                             false, false, false, false, false};
    for (std::size_t index = 0; index < deck_order.size(); ++index) {
        const int card = deck_order[index];
        require(card >= 0 && card < kCardCount,
                "serialized deck contains an invalid card");
        require(!seen[card], "serialized deck contains a duplicate card");
        require((remaining_mask_ & (std::uint16_t(1u) << card)) != 0,
                "serialized deck contains a consumed card");
        seen[card] = true;
    }
    deck_order_ = deck_order;
    public_copy_ = public_copy;
}

std::string GameState::canonical_signature() const {
    const PublicState state = canonical();
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    output << static_cast<int>(state.terminated) << ':'
           << static_cast<int>(state.round_index) << ':'
           << static_cast<int>(state.underground_count) << ':'
           << static_cast<int>(state.draw_count) << ':'
           << state.remaining_mask;
    output << ":o";
    for (int color = 0; color < kColorCount; ++color) {
        output << static_cast<int>(state.order[color]);
    }
    for (int color = 0; color < kColorCount; ++color) {
        output << ":s" << std::setw(14) << state.line_station_masks[color];
        output << ":e";
        for (int word = 0; word < 3; ++word) {
            output << std::setw(16) << state.line_edge_masks[color].words[word];
        }
    }
    return output.str();
}

std::array<int, 6> GameState::partial_score_components() const {
    std::array<int, 6> result = {{
        route_total_ + circle_route_bonus(), thames_total_,
        partial_tourist_visits_, partial_tourist_points_, interchange_total_,
        popcount_u64(completed_objective_mask_) * 10,
    }};
    return result;
}

int GameState::current_total() const {
    return route_total_ + circle_route_bonus() + thames_total_
        + partial_tourist_points_ + interchange_total_
        + popcount_u64(completed_objective_mask_) * 10;
}

void GameState::write_features(float* destination, std::size_t capacity) const {
    require(destination != 0, "feature destination is required");
    require(capacity >= static_cast<std::size_t>(kObservationDim),
            "feature destination is too small");
    require(!has_pending_, "features require a pending-free afterstate");

    std::size_t cursor = 0;
    for (int color = 0; color < kColorCount; ++color) {
        for (int edge = 0; edge < kEdgeCount; ++edge) {
            destination[cursor++] = lines_[color].edge_mask.test(edge) ? 1.0f : 0.0f;
        }
    }
    for (int color = 0; color < kColorCount; ++color) {
        for (int station = 0; station < kStationCount; ++station) {
            destination[cursor++] =
                (lines_[color].station_mask & (std::uint64_t(1) << station))
                ? 1.0f : 0.0f;
        }
    }
    for (int card = 0; card < kCardCount; ++card) {
        destination[cursor++] =
            (remaining_mask_ & (std::uint16_t(1) << card)) ? 1.0f : 0.0f;
    }
    for (int position = 0; position < kColorCount; ++position) {
        for (int color = 0; color < kColorCount; ++color) {
            destination[cursor++] = order_[position] == color ? 1.0f : 0.0f;
        }
    }
    for (int round = 0; round < kColorCount; ++round) {
        destination[cursor++] = round_index_ == round ? 1.0f : 0.0f;
    }
    const int active_color = order_[round_index_];
    for (int color = 0; color < kColorCount; ++color) {
        destination[cursor++] = active_color == color ? 1.0f : 0.0f;
    }
    for (int count = 0; count < 6; ++count) {
        destination[cursor++] = underground_count_ == count ? 1.0f : 0.0f;
    }
    for (int count = 0; count <= kCardCount; ++count) {
        destination[cursor++] = draw_count_ == count ? 1.0f : 0.0f;
    }
    destination[cursor++] = terminated() ? 1.0f : 0.0f;

    for (int color = 0; color < kColorCount; ++color) {
        const LineMetrics& metrics = metrics_[color];
        for (int district = 0; district < kDistrictCount; ++district) {
            destination[cursor++] = static_cast<float>(
                static_cast<double>(metrics.station_counts[district]) / 13.0);
        }
        for (int district = 0; district < kDistrictCount; ++district) {
            destination[cursor++] =
                (metrics.district_mask & (std::uint16_t(1) << district))
                ? 1.0f : 0.0f;
        }
        destination[cursor++] = static_cast<float>(
            static_cast<double>(popcount_u64(metrics.district_mask)) / 13.0);
        destination[cursor++] = static_cast<float>(
            static_cast<double>(metrics.max_stations) / 13.0);
        destination[cursor++] = static_cast<float>(
            static_cast<double>(metrics.route) / 169.0);
        destination[cursor++] = static_cast<float>(
            static_cast<double>(metrics.thames_crossings * 2) / 20.0);
        destination[cursor++] = static_cast<float>(
            static_cast<double>(metrics.tourist_visits) / 5.0);
    }

    destination[cursor++] = static_cast<float>(
        static_cast<double>(route_total_) / 676.0);
    destination[cursor++] = static_cast<float>(
        static_cast<double>(thames_total_) / 80.0);
    destination[cursor++] = static_cast<float>(
        static_cast<double>(partial_tourist_visits_) / 20.0);
    destination[cursor++] = static_cast<float>(
        static_cast<double>(partial_tourist_points_) / 25.0);
    const int tourist_tier = std::min(partial_tourist_visits_, 10);
    for (int tier = 0; tier < 11; ++tier) {
        destination[cursor++] = tourist_tier == tier ? 1.0f : 0.0f;
    }
    for (int count = 0; count <= kColorCount; ++count) {
        destination[cursor++] = static_cast<float>(
            static_cast<double>(interchange_counts_[count]) / 53.0);
    }
    destination[cursor++] = static_cast<float>(
        static_cast<double>(interchange_total_) / 477.0);
    destination[cursor++] = static_cast<float>(
        static_cast<double>(current_total()) / 1300.0);
    const int next_tourist_gain =
        kTouristTrack[std::min(partial_tourist_visits_ + 1, 10)]
        - kTouristTrack[tourist_tier];
    destination[cursor++] = static_cast<float>(
        static_cast<double>(next_tourist_gain) / 4.0);
    for (int count = 0; count < kColorCount; ++count) {
        const int unit_gain = kInterchangeTrack[count + 1]
            - kInterchangeTrack[count];
        destination[cursor++] = static_cast<float>(
            static_cast<double>(unit_gain) / 9.0);
    }
    for (int count = 0; count < kColorCount; ++count) {
        const int unit_gain = kInterchangeTrack[count + 1]
            - kInterchangeTrack[count];
        destination[cursor++] = static_cast<float>(
            static_cast<double>(interchange_counts_[count] * unit_gain) / 477.0);
    }
    require(cursor == static_cast<std::size_t>(kObservationDim),
            "feature schema dimension mismatch");
}

FinalScore GameState::final_score() const {
    FinalScore result;
    for (std::size_t i = 0; i < round_scores_.size(); ++i) {
        result.line_total += round_scores_[i].total;
    }
    result.tourist_visits = partial_tourist_visits_;
    if (round_scores_.size() < static_cast<std::size_t>(kColorCount)) {
        int completed_visits = 0;
        for (std::size_t i = 0; i < round_scores_.size(); ++i) {
            const int color = order_[i];
            completed_visits += metrics_[color].tourist_visits;
        }
        result.tourist_visits = completed_visits;
    }
    result.tourist_bonus = kTouristTrack[
        std::min(result.tourist_visits, 10)];
    result.two_line_stations = interchange_counts_[2];
    result.three_line_stations = interchange_counts_[3];
    result.four_line_stations = interchange_counts_[4];
    result.interchange_bonus = interchange_total_;
    result.objectives_completed = popcount_u64(completed_objective_mask_);
    result.objective_bonus = result.objectives_completed * 10;
    result.total = result.line_total + result.tourist_bonus
        + result.interchange_bonus + result.objective_bonus;
    return result;
}

Candidate::Candidate(int index, const Action& selected, int value,
                     const GameState& child)
    : action_index(index), action(selected), reward(value), afterstate(child) {}

ChanceOutcome::ChanceOutcome(double probability_value, const GameState& child)
    : probability(probability_value), state(child) {}

}  // namespace native
}  // namespace next_station
