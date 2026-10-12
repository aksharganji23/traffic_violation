def crossed_line(previous_y, current_y, stop_line_y):

    return (
        previous_y < stop_line_y
        and
        current_y >= stop_line_y
    )


def check_red_light(
    vehicle_history,
    stop_line_y,
    signal_is_red
):

    violations = []

    if not signal_is_red:
        return violations

    for vehicle_id, history in vehicle_history.items():

        if len(history) < 2:
            continue

        previous_y = history[-2][1]
        current_y = history[-1][1]

        if crossed_line(
            previous_y,
            current_y,
            stop_line_y
        ):

            violations.append(vehicle_id)

    return violations