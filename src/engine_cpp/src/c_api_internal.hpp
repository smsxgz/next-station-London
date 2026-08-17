#ifndef NEXT_STATION_NATIVE_C_API_INTERNAL_HPP
#define NEXT_STATION_NATIVE_C_API_INTERNAL_HPP

#include "next_station/c_api.h"
#include "next_station/engine.hpp"

namespace next_station {
namespace native {

GameState& game_from_c_handle(ns_game_handle handle);

}  // namespace native
}  // namespace next_station

#endif  // NEXT_STATION_NATIVE_C_API_INTERNAL_HPP
