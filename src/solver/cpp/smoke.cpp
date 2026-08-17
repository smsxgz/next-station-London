#include "search.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        next_station::native::GameState game(1234);
        game.draw_known_cards(std::vector<int>(1, 0));

        const next_station::solver_native::LookaheadResult lookahead =
            next_station::solver_native::rank_lookahead(game, 2, true);
        if (lookahead.estimates.empty()
            || lookahead.estimates[0].action.edge_id != -1
            || lookahead.stats.decision_nodes < 1) {
            throw std::runtime_error("native lookahead smoke check failed");
        }

        const next_station::solver_native::MCTSResult mcts =
            next_station::solver_native::rank_mcts(game, 32, 22.5, 9876);
        std::int64_t visits = 0;
        for (std::size_t index = 0; index < mcts.estimates.size(); ++index) {
            visits += mcts.estimates[index].visits;
        }
        if (visits != 32 || mcts.stats.simulations != 32
            || !std::isfinite(mcts.stats.elapsed_seconds)) {
            throw std::runtime_error("native MCTS smoke check failed");
        }
        const next_station::solver_native::MCTSResult lookahead_mcts =
            next_station::solver_native::rank_mcts(
                game,
                8,
                22.5,
                9876,
                next_station::solver_native::MCTSRolloutPolicy::Lookahead2);
        visits = 0;
        for (std::size_t index = 0;
             index < lookahead_mcts.estimates.size(); ++index) {
            visits += lookahead_mcts.estimates[index].visits;
        }
        if (visits != 8 || lookahead_mcts.stats.simulations != 8
            || !std::isfinite(lookahead_mcts.stats.elapsed_seconds)) {
            throw std::runtime_error("native lookahead-rollout MCTS smoke check failed");
        }
        const next_station::solver_native::MCTSResult random_mcts =
            next_station::solver_native::rank_mcts(
                game,
                8,
                22.5,
                9876,
                next_station::solver_native::MCTSRolloutPolicy::SimpleRandom);
        visits = 0;
        for (std::size_t index = 0; index < random_mcts.estimates.size(); ++index) {
            visits += random_mcts.estimates[index].visits;
        }
        if (visits != 8 || random_mcts.stats.simulations != 8
            || !std::isfinite(random_mcts.stats.elapsed_seconds)) {
            throw std::runtime_error("native random-rollout MCTS smoke check failed");
        }
        std::cout << "next_station_solver_check: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
