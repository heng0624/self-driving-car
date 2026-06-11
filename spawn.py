import random

def spawn_vehicle(world):

    bp_lib = world.get_blueprint_library()

    vehicle_bp = bp_lib.filter("model3")[0]

    spawn_point = random.choice(
        world.get_map().get_spawn_points()
    )

    vehicle = world.spawn_actor(
        vehicle_bp,
        spawn_point
    )

    return vehicle