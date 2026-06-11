import numpy as np

def build_state(
    image,
    speed,
    lane_offset
):

    state = {
        "image": image,
        "speed": speed,
        "lane_offset": lane_offset
    }

    return state