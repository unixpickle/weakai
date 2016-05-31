# minimax

Minimax is an algorithm used to make decisions in games where opponents have opposite goals. This demo uses it to play checkers. Checkers is probably a bad example, since there are simpler algorithms for it, but nevertheless I think it will help me learn about minimax.

# Enhancements

This implementation of Minimax uses two techniques to improve its performance. First, it uses the "alpha-beta" technique to avoid exploring nodes that cannot yield a better/worse outcome than the existing best/worst option. Second, it uses iterative deepening to search to a dynamic depth on each move. This ensures that it spends roughly one second per iteration, regardless of how deep it ends up going.
