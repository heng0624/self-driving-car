def calculate_reward(
    collision,
    speed,
    lane_offset,
    red_light
):

    reward = 0

    reward += speed * 0.05

    reward -= abs(lane_offset)

    if collision:
        reward -= 100

    if red_light:
        reward -= 50

    return reward