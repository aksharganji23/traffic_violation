def detect_wrong_way(
    history,
    expected_direction="left_to_right"
):

    if len(history) < 5:
        return False

    first_x = history[0][0]
    last_x = history[-1][0]

    movement = last_x - first_x

    if expected_direction == "left_to_right":

        return movement < -20

    elif expected_direction == "right_to_left":

        return movement > 20

    return False