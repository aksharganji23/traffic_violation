def point_inside_box(point, box):

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def detect_triple_riding(
    motorcycles,
    persons
):

    violations = []

    for motorcycle in motorcycles:

        bike_box = motorcycle["bbox"]

        rider_count = 0

        for person in persons:

            person_center = person["centroid"]

            if point_inside_box(
                person_center,
                bike_box
            ):

                rider_count += 1

        if rider_count > 2:

            violations.append({
                "bbox": bike_box,
                "riders": rider_count
            })

    return violations