import robosuite as suite
from robosuite.wrappers.tactile_wrapper import TactileWrapper
from robosuite import load_controller_config


def get_door_env(rank: int = 0, sparse: bool = False, state_type="vision_and_touch", **kwargs):
    robots = ["PandaTactile"]
    placement_initializer = None
    init_qpos = [-0.073, 0.016, -0.392, -2.502, 0.240, 2.676, 0.189]
    env_config = kwargs.copy()
    config = load_controller_config(default_controller="OSC_POSE")
    env_config["robot_configs"] = [{"initial_qpos": init_qpos}]
    env_config["initialization_noise"] = None
    env = TactileWrapper(
        suite.make(
            "Door",
            robots=robots,  # use PandaTactile robot
            use_camera_obs=True,  # use pixel observations
            use_object_obs=False,
            has_offscreen_renderer=True,  # needed for pixel obs
            has_renderer=False,  # not needed due to offscreen rendering
            reward_shaping=bool(1 - sparse),  # use dense rewards
            camera_names="agentview",
            horizon=300,
            controller_configs=config,
            placement_initializer=placement_initializer,
            camera_heights=84,
            camera_widths=84,
            **env_config,
        ),
        env_id=rank,
        state_type=state_type,
    )
    return env
