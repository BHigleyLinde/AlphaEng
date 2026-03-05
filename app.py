import streamlit as st
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None


PRECOMPUTED_FIRST_GUESS = {
    (1, 100): 75,
}

PRECOMPUTED_FOLLOWING_GUESSES = {
    (1, 100, 75): (25, 50),
}


def win_probability(my_guess, other_guesses, low=1, high=100):
    """
    Probability that `my_guess` wins when target is uniformly random in [low, high].
    Distance is on a number line (no wrap-around).
    Ties are split evenly among tied players.
    """
    return _win_probability_cached(my_guess, tuple(sorted(other_guesses)), low, high)


@lru_cache(maxsize=None)
def _range_distance_cache(low, high):
    if np is None:
        return None
    values = np.arange(low, high + 1, dtype=np.int32)
    distances = np.abs(values[:, None] - values[None, :])
    return values, distances


@lru_cache(maxsize=None)
def _win_probability_cached(my_guess, other_guesses, low, high):
    if np is not None:
        _, distances = _range_distance_cache(low, high)
        my_idx = my_guess - low
        all_indices = [my_idx] + [g - low for g in other_guesses]
        stacked = distances[all_indices]

        best = stacked.min(axis=0)
        winners = (stacked == best).sum(axis=0)
        wins = np.where(stacked[0] == best, 1.0 / winners, 0.0).sum()
        return float(wins / (high - low + 1))

    all_guesses = (my_guess,) + other_guesses
    my_index = 0
    wins = 0.0

    for target in range(low, high + 1):
        distances = [abs(target - g) for g in all_guesses]
        best_dist = min(distances)
        winners = [i for i, d in enumerate(distances) if d == best_dist]

        if my_index in winners:
            wins += 1 / len(winners)

    return wins / (high - low + 1)


def _third_payoff_vector(first_guess, second_guess, low, high):
    _, distances = _range_distance_cache(low, high)
    first_idx = first_guess - low
    second_idx = second_guess - low

    my_distances = distances
    first_distances = distances[first_idx][None, :]
    second_distances = distances[second_idx][None, :]

    best = np.minimum(my_distances, np.minimum(first_distances, second_distances))
    me_wins = my_distances == best
    winners = (
        me_wins.astype(np.int8)
        + (first_distances == best).astype(np.int8)
        + (second_distances == best).astype(np.int8)
    )
    return (me_wins / winners).mean(axis=1)


def midpoint_of_largest_gap(existing_guesses, low=1, high=100):
    """
    Midpoint of the largest gap among [low], existing guesses, and [high].
    Used as a tie-break target so equal-optimal choices prefer a central pick.
    """
    anchors = [low] + sorted(existing_guesses) + [high]
    best_gap = -1
    best_mid = (low + high) / 2.0

    for left, right in zip(anchors, anchors[1:]):
        gap = right - left
        mid = (left + right) / 2.0
        if gap > best_gap:
            best_gap = gap
            best_mid = mid

    return best_mid


def break_tie_by_gap_midpoint(candidates, existing_guesses, low=1, high=100):
    """Pick candidate closest to midpoint of largest gap; then closest to global center."""
    if len(candidates) == 1:
        return candidates[0]

    target = midpoint_of_largest_gap(existing_guesses, low, high)
    center = (low + high) / 2.0
    return min(candidates, key=lambda x: (abs(x - target), abs(x - center), x))


def legal_candidates(prior_guesses, low=1, high=100):
    blocked = set(prior_guesses)
    candidates = [x for x in range(low, high + 1) if x not in blocked]
    if not candidates:
        raise ValueError("No legal guesses are available within the selected range.")
    return candidates


def best_guess_third(prior_guesses, low=1, high=100):
    """Best response for player 3 against two existing guesses."""
    if len(prior_guesses) != 2:
        raise ValueError("Position 3 requires exactly two prior guesses.")
    if len(set(prior_guesses)) != len(prior_guesses):
        raise ValueError("Prior guesses must be unique.")
    a, b = sorted(prior_guesses)  # order does not affect distance outcome
    return _best_guess_third_cached(a, b, low, high)


@lru_cache(maxsize=None)
def _best_guess_third_cached(first_guess, second_guess, low=1, high=100):
    prior_guesses = [first_guess, second_guess]

    if np is not None:
        candidates = legal_candidates(prior_guesses, low, high)
        candidate_indices = np.array([g - low for g in candidates], dtype=np.int32)
        payoffs = _third_payoff_vector(first_guess, second_guess, low, high)
        candidate_payoffs = payoffs[candidate_indices]

        best_p = float(candidate_payoffs.max())
        best_indices = candidate_indices[np.isclose(candidate_payoffs, best_p, rtol=0.0, atol=1e-12)]
        best_candidates = [int(idx + low) for idx in best_indices.tolist()]
        return break_tie_by_gap_midpoint(best_candidates, prior_guesses, low, high)

    candidates = legal_candidates(prior_guesses, low, high)
    best_candidates = []
    best_p = -1.0
    eps = 1e-12

    for x in candidates:
        p = win_probability(x, prior_guesses, low, high)
        if p > best_p + eps:
            best_p = p
            best_candidates = [x]
        elif abs(p - best_p) <= eps:
            best_candidates.append(x)

    return break_tie_by_gap_midpoint(best_candidates, prior_guesses, low, high)


@lru_cache(maxsize=None)
def best_guess_second(first_guess, low=1, high=100):
    """
    Choose player 2 guess assuming player 3 will respond optimally for themselves.
    Returns the player-2 guess that maximizes player-2 final win probability.
    """
    candidates = legal_candidates([first_guess], low, high)
    best_candidates = []
    best_second_p = -1.0
    eps = 1e-12

    for second_guess in candidates:
        third_guess = best_guess_third([first_guess, second_guess], low, high)
        p_second = win_probability(second_guess, [first_guess, third_guess], low, high)

        if p_second > best_second_p + eps:
            best_second_p = p_second
            best_candidates = [second_guess]
        elif abs(p_second - best_second_p) <= eps:
            best_candidates.append(second_guess)

    return break_tie_by_gap_midpoint(best_candidates, [first_guess], low, high)


@lru_cache(maxsize=None)
def optimal_following_guesses(first_guess, low=1, high=100):
    """Optimal (P2, P3) responses after a first guess."""
    precomputed = PRECOMPUTED_FOLLOWING_GUESSES.get((low, high, first_guess))
    if precomputed is not None:
        return precomputed

    second_guess = best_guess_second(first_guess, low, high)
    third_guess = best_guess_third([first_guess, second_guess], low, high)
    return second_guess, third_guess


@lru_cache(maxsize=None)
def best_guess_first(low=1, high=100):
    """
    Choose player 1 guess anticipating optimal play from player 2 and player 3.
    """
    precomputed = PRECOMPUTED_FIRST_GUESS.get((low, high))
    if precomputed is not None:
        return precomputed

    candidates = list(range(low, high + 1))
    best_candidates = []
    best_first_p = -1.0
    eps = 1e-12

    for first_guess in candidates:
        second_guess, third_guess = optimal_following_guesses(first_guess, low, high)
        p_first = win_probability(first_guess, [second_guess, third_guess], low, high)

        if p_first > best_first_p + eps:
            best_first_p = p_first
            best_candidates = [first_guess]
        elif abs(p_first - best_first_p) <= eps:
            best_candidates.append(first_guess)

    return break_tie_by_gap_midpoint(best_candidates, [], low, high)


def best_guess(position, prior_guesses=None, low=1, high=100):
    if prior_guesses is None:
        prior_guesses = []

    if position not in (1, 2, 3):
        raise ValueError("Position must be 1, 2, or 3.")
    if len(prior_guesses) != position - 1:
        raise ValueError(f"Position {position} requires {position - 1} prior guess(es).")
    if len(set(prior_guesses)) != len(prior_guesses):
        raise ValueError("Prior guesses must be unique.")
    if any(g < low or g > high for g in prior_guesses):
        raise ValueError(f"All prior guesses must be between {low} and {high}.")

    if position == 1:
        return best_guess_first(low, high)
    if position == 2:
        return best_guess_second(prior_guesses[0], low, high)
    return best_guess_third(prior_guesses, low, high)


st.set_page_config(page_title="Best Guess", page_icon="🎯")
st.title("Best Guess")
st.caption("Pick the best number on 1-100 using backward-induction optimal-play logic.")

low, high = 1, 100
position = st.selectbox("Your position", [1, 2, 3], index=0)

if position == 1:
    with st.spinner("Computing follower-aware recommendation..."):
        recommended_guess = best_guess(1, low=low, high=high)

    st.success(f"Recommended guess: {recommended_guess}")

    actual_guess = int(
        st.number_input(
            "Your actual guess",
            min_value=low,
            max_value=high,
            value=recommended_guess,
            step=1,
            key="p1_actual_guess",
        )
    )
    if actual_guess != recommended_guess:
        st.caption(f"Note: Recommended guess was {recommended_guess}.")

    optimal_second, optimal_third = optimal_following_guesses(actual_guess, low, high)

    prob_optimal = win_probability(actual_guess, [optimal_second, optimal_third], low, high)
    st.write(
        f"Expected probability for your guess vs optimal follower play (P2={optimal_second}, P3={optimal_third}): "
        f"{prob_optimal:.4f} ({prob_optimal * 100:.2f}%)"
    )

    second = int(
        st.number_input(
            "Actual second player's guess",
            min_value=low,
            max_value=high,
            value=optimal_second,
            step=1,
            key="p1_actual_second",
        )
    )
    third = int(
        st.number_input(
            "Actual third player's guess",
            min_value=low,
            max_value=high,
            value=optimal_third,
            step=1,
            key="p1_actual_third",
        )
    )

    if len({actual_guess, second, third}) < 3:
        st.error("All three guesses must be different.")
    else:
        prob_actual = win_probability(actual_guess, [second, third], low, high)
        st.write(
            f"Your probability of winning vs those actual guesses: "
            f"{prob_actual:.4f} ({prob_actual * 100:.2f}%)"
        )

elif position == 2:
    first = int(
        st.number_input(
            "First player's guess",
            min_value=low,
            max_value=high,
            value=25,
            step=1,
            key="p2_first_guess",
        )
    )
    recommended_guess = best_guess(2, [first], low, high)
    st.success(f"Recommended guess: {recommended_guess}")

    actual_guess = int(
        st.number_input(
            "Your actual guess",
            min_value=low,
            max_value=high,
            value=recommended_guess,
            step=1,
            key="p2_actual_guess",
        )
    )
    if actual_guess != recommended_guess:
        st.caption(f"Note: Recommended guess was {recommended_guess}.")

    if actual_guess == first:
        st.error("Your guess cannot match the first player's guess.")
        st.stop()

    optimal_third = best_guess_third([first, actual_guess], low, high)
    prob_vs_optimal = win_probability(actual_guess, [first, optimal_third], low, high)
    st.write(
        f"Expected probability for your guess vs optimal third-player response ({optimal_third}): "
        f"{prob_vs_optimal:.4f} ({prob_vs_optimal * 100:.2f}%)"
    )

    third = int(
        st.number_input(
            "Actual third player's guess",
            min_value=low,
            max_value=high,
            value=optimal_third,
            step=1,
            key="p2_actual_third",
        )
    )

    if third in {first, actual_guess}:
        st.error("Third player's guess must be different from both existing guesses.")
    else:
        prob_actual = win_probability(actual_guess, [first, third], low, high)
        st.write(f"Your probability of winning vs that actual guess: {prob_actual:.4f} ({prob_actual * 100:.2f}%)")

else:
    first = int(
        st.number_input(
            "First player's guess",
            min_value=low,
            max_value=high,
            value=25,
            step=1,
            key="p3_first_guess",
        )
    )
    second = int(
        st.number_input(
            "Second player's guess",
            min_value=low,
            max_value=high,
            value=76,
            step=1,
            key="p3_second_guess",
        )
    )

    if first == second:
        st.error("First and second player guesses must be different.")
        st.stop()

    recommended_guess = best_guess(3, [first, second], low, high)
    st.success(f"Recommended guess: {recommended_guess}")

    actual_guess = int(
        st.number_input(
            "Your actual guess",
            min_value=low,
            max_value=high,
            value=recommended_guess,
            step=1,
            key="p3_actual_guess",
        )
    )
    if actual_guess != recommended_guess:
        st.caption(f"Note: Recommended guess was {recommended_guess}.")

    if actual_guess in {first, second}:
        st.error("Your guess must be different from the first and second player guesses.")
    else:
        prob = win_probability(actual_guess, [first, second], low, high)
        st.write(f"Your probability of winning: {prob:.4f} ({prob * 100:.2f}%)")
