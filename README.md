# Sokoban solver

Push-optimal Sokoban solver written from scratch in TypeScript for the
make.fun competition `sokoban-solver-01-run-20261121`.

Approach: a level decomposes into independent rooms off a 1-wide hallway,
and boxes never leave their room, so each room is solved on its own with a
push-based A* (min-cost box-to-goal matching heuristic, static dead squares,
frozen-box and 2x2 deadlock pruning, player-region canonicalisation), with a
greedy best-first fallback. Room solutions are stitched with shortest walks
through the hallway. All budgets are counted in expansions, never wall time,
so two runs write identical bytes.

Practice set: 1,000,000 (all five instances solved in exactly P* pushes:
13 / 16 / 29 / 47 / 60). House baseline plain-astar scores 200,000 there.
