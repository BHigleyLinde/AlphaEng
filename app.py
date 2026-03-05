import streamlit as st


def win_probability(my_guess, other_guesses, low=1, high=100):
    """
    Probability that `my_guess` wins when target is uniformly random in [low, high].
    Distance is on a number line (no wrap-around).
    Ties are split evenly among tied players.
    """
    all_guesses = [my_guess] + other_guesses
    my_index = 0
    wins = 0.0

    for target in range(low, high + 1):
        distances = [abs(target - g) for g in all_guesses]
        best_dist = min(distances)
        winners = [i for i, d in enumerate(distances) if d == best_dist]

        if my_index in winners:
            wins += 1 / len(winners)

    return wins / (high - low + 1)


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


def best_guess_third(prior_guesses, low=1, high=100):
    """Best response for player 3 against two existing guesses."""
    candidates = list(range(low, high + 1))
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


def best_guess_second(first_guess, low=1, high=100):
    """
    Choose player 2 guess assuming player 3 will respond optimally for themselves.
    Returns the player-2 guess that maximizes player-2 final win probability.
    """
    candidates = list(range(low, high + 1))
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


def best_guess(position, prior_guesses=None, low=1, high=100):
    if prior_guesses is None:
        prior_guesses = []

    if position not in (1, 2, 3):
        raise ValueError("Position must be 1, 2, or 3.")
    if len(prior_guesses) != position - 1:
        raise ValueError(f"Position {position} requires {position - 1} prior guess(es).")
    if any(g < low or g > high for g in prior_guesses):
        raise ValueError(f"All prior guesses must be between {low} and {high}.")

    if position == 1:
        return (low + high) // 2

    if position == 2:
        return best_guess_second(prior_guesses[0], low, high)

    return best_guess_third(prior_guesses, low, high)


st.set_page_config(page_title="Best Guess", page_icon="🎯")
st.title("Best Guess")
st.caption("Pick the best number on 1-100 using optimal-play logic with a midpoint tie-breaker.")

low, high = 1, 100
position = st.selectbox("Your position", [1, 2, 3], index=0)

if position == 1:
    guess = best_guess(1, low=low, high=high)
    st.success(f"Recommended guess: {guess}")

    second = int(st.number_input("Second player's guess", min_value=low, max_value=high, value=75, step=1))
    third = int(st.number_input("Third player's guess", min_value=low, max_value=high, value=25, step=1))
    prob = win_probability(guess, [second, third], low, high)
    st.write(f"Your probability of winning: {prob:.4f} ({prob * 100:.2f}%)")

elif position == 2:
    first = int(st.number_input("First player's guess", min_value=low, max_value=high, value=25, step=1))
    guess = best_guess(2, [first], low, high)
    st.success(f"Recommended guess: {guess}")

    optimal_third = best_guess_third([first, guess], low, high)
    prob_vs_optimal = win_probability(guess, [first, optimal_third], low, high)
    st.write(
        f"Expected probability vs optimal third-player response ({optimal_third}): "
        f"{prob_vs_optimal:.4f} ({prob_vs_optimal * 100:.2f}%)"
    )

    third = int(
        st.number_input(
            "Actual third player's guess",
            min_value=low,
            max_value=high,
            value=optimal_third,
            step=1,
        )
    )
    prob_actual = win_probability(guess, [first, third], low, high)
    st.write(f"Your probability of winning vs that actual guess: {prob_actual:.4f} ({prob_actual * 100:.2f}%)")

else:
    first = int(st.number_input("First player's guess", min_value=low, max_value=high, value=25, step=1))
    second = int(st.number_input("Second player's guess", min_value=low, max_value=high, value=76, step=1))
    guess = best_guess(3, [first, second], low, high)
    st.success(f"Recommended guess: {guess}")

    prob = win_probability(guess, [first, second], low, high)
    st.write(f"Your probability of winning: {prob:.4f} ({prob * 100:.2f}%)")
