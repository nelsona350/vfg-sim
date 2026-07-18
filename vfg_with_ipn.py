from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


Vector3 = NDArray[np.float64]
SMALL_NUMBER = 1.0e-9
LAUNCH_RECOMPUTE_INTERVAL_S = 1.0
PRELAUNCH_TARGET_PROPAGATION_RATE_HZ = 10.0


@dataclass
class GuidanceConfig:
    waypoints_ned: NDArray[np.float64]
    initial_position_ned: Vector3
    desired_flight_speed: float
    max_waypoint_intercept_angle_rad: float
    convergence_time_constant: float
    waypoint_plane_tolerance: float


@dataclass
class IntegratedPNConfig:
    terminal_range: float
    horizontal_navigation_constant: float
    vertical_navigation_constant: float


@dataclass
class SimulationConfig:
    version: str
    scenario_title: str
    classification: str
    dataset_name: str

    vehicle_launch_time: float
    vehicle: GuidanceConfig
    target: GuidanceConfig
    integrated_pn: IntegratedPNConfig

    time_step: float
    simulation_duration: float
    output_file: Path


@dataclass
class EntityState:
    position_ned: Vector3
    waypoint_index: int = 1
    velocity_ned: Vector3 | None = None
    terminal_guidance_active: bool = False
    previous_los_azimuth_rad: float | None = None
    previous_los_elevation_rad: float | None = None
    commanded_heading_rad: float | None = None
    commanded_flight_path_angle_rad: float | None = None


@dataclass
class EntityGuidanceResult:
    velocity_ned: Vector3
    waypoint_index: int


@dataclass
class PrelaunchTargetPropagation:
    time_history: NDArray[np.float64]
    target_position_history_ned: NDArray[np.float64]
    target_velocity_history_ned: NDArray[np.float64]


@dataclass
class SimulationResult:
    time_history: NDArray[np.float64]
    vehicle_position_history_ned: NDArray[np.float64]
    vehicle_velocity_history_ned: NDArray[np.float64]
    target_position_history_ned: NDArray[np.float64]
    target_velocity_history_ned: NDArray[np.float64]
    range_history: NDArray[np.float64]
    terminal_guidance_history: NDArray[np.float64]
    poca_time: float
    poca_range: float
    terminal_guidance_activation_time: float | None
    target_time_to_vehicle_final_waypoint_poca: float
    vehicle_time_to_final_waypoint: float
    vehicle_launch_time: float
    prelaunch_target_propagations: list[PrelaunchTargetPropagation]


def _as_vector3(values: Sequence[float], name: str) -> Vector3:
    vector = np.asarray(values, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly three values; got shape {vector.shape}."
        )

    return vector


def _validate_guidance_config(config: GuidanceConfig, name: str) -> None:
    if config.waypoints_ned.ndim != 2 or config.waypoints_ned.shape[1] != 3:
        raise ValueError(f"{name}.waypoints_ned must have shape (N, 3).")

    if len(config.waypoints_ned) < 2:
        raise ValueError(f"{name} requires at least two waypoints.")

    if config.desired_flight_speed <= 0.0:
        raise ValueError(f"{name}.desired_flight_speed must be positive.")

    if not 0.0 < config.max_waypoint_intercept_angle_rad < math.pi / 2.0:
        raise ValueError(
            f"{name}.max_waypoint_intercept_angle must be between 0 and 90 degrees."
        )

    if config.convergence_time_constant <= 0.0:
        raise ValueError(
            f"{name}.convergence_time_constant must be positive."
        )

    if config.waypoint_plane_tolerance < 0.0:
        raise ValueError(
            f"{name}.waypoint_plane_tolerance must be nonnegative."
        )


def _validate_config(config: SimulationConfig) -> None:
    _validate_guidance_config(config.vehicle, "vehicle")
    _validate_guidance_config(config.target, "target")

    if config.time_step <= 0.0:
        raise ValueError("time_step must be positive.")

    if config.simulation_duration <= 0.0:
        raise ValueError("simulation_duration must be positive.")

    if config.vehicle_launch_time >= config.simulation_duration:
        raise ValueError(
            "vehicle.launch_time_s must be less than simulation_duration."
        )

    if config.integrated_pn.terminal_range <= 0.0:
        raise ValueError("integrated_pn.terminal_range must be positive.")

    if config.integrated_pn.horizontal_navigation_constant <= 0.0:
        raise ValueError(
            "integrated_pn.horizontal_navigation_constant must be positive."
        )

    if config.integrated_pn.vertical_navigation_constant <= 0.0:
        raise ValueError(
            "integrated_pn.vertical_navigation_constant must be positive."
        )



def _load_json_guidance(
    data: dict[str, Any],
    entity_name: str,
) -> GuidanceConfig:
    guidance = data["guidance"]
    initial_position_key = "initial_position_ned"

    return GuidanceConfig(
        waypoints_ned=np.asarray(data["waypoints_ned"], dtype=np.float64),
        initial_position_ned=_as_vector3(
            data[initial_position_key],
            f"{entity_name}.{initial_position_key}",
        ),
        desired_flight_speed=float(guidance["desired_flight_speed_mps"]),
        max_waypoint_intercept_angle_rad=math.radians(
            float(guidance["max_waypoint_intercept_angle_deg"])
        ),
        convergence_time_constant=float(
            guidance.get("convergence_time_constant_s", 3.0)
        ),
        waypoint_plane_tolerance=float(
            guidance.get("waypoint_plane_tolerance_m", 0.0)
        ),
    )


def load_json_config(input_path: Path) -> SimulationConfig:
    with input_path.open("r", encoding="utf-8") as input_file:
        data: dict[str, Any] = json.load(input_file)

    metadata = data.get("metadata", {})
    simulation = data["simulation"]
    integrated_pn = data["vehicle"]["integrated_proportional_navigation"]


    config = SimulationConfig(
        version=str(metadata.get("version", "1.0")),
        scenario_title=str(metadata.get("scenario_title", "TBolt")),
        classification=str(metadata.get("classification", "UNCLASSIFIED")),
        dataset_name=str(metadata.get("dataset_name", "Thunderbolt")),
        vehicle_launch_time=float(data["vehicle"].get("launch_time_s", 0.0)),
        vehicle=_load_json_guidance(data["vehicle"], "vehicle"),
        target=_load_json_guidance(data["target"], "target"),
        integrated_pn=IntegratedPNConfig(
            terminal_range=float(integrated_pn["terminal_range_m"]),
            horizontal_navigation_constant=float(
                integrated_pn.get(
                    "horizontal_navigation_constant",
                    integrated_pn.get("navigation_constant", 3.0),
                )
            ),
            vertical_navigation_constant=float(
                integrated_pn.get(
                    "vertical_navigation_constant",
                    integrated_pn.get("navigation_constant", 3.0),
                )
            ),
        ),
        time_step=float(simulation["time_step_s"]),
        simulation_duration=float(simulation["duration_s"]),
        output_file=Path(simulation.get("output_file", "vfg_state.out")),
    )

    config.vehicle_launch_time = compute_synchronized_vehicle_launch_time(
        config
    )

    _validate_config(config)
    return config


def compute_constant_heading_velocity_ned(
    config: GuidanceConfig,
) -> Vector3:
    """Return constant-velocity target propagation along its first leg."""
    path_vector_ned = config.waypoints_ned[0] - config.initial_position_ned
    path_length = float(np.linalg.norm(path_vector_ned))

    if path_length <= SMALL_NUMBER:
        raise ValueError(
            "Cannot determine constant heading from an initial position "
            "coincident with the first waypoint."
        )

    return config.desired_flight_speed * path_vector_ned / path_length


def align_vehicle_penultimate_waypoint_with_target_approach(
    config: SimulationConfig,
    target_velocity_ned: Vector3,
) -> None:
    """
    Move the vehicle's next-to-last waypoint so its final leg approaches the
    predicted intercept point anti-parallel to the target's propagation.

    The vehicle's final waypoint is the predicted intercept point used by the
    launch synchronization logic.  Keeping the final leg length unchanged while
    rotating it about that final waypoint preserves the configured terminal
    geometry scale, but makes the vehicle's direction from the next-to-last
    waypoint to the final waypoint oppose the target's current pre-launch
    propagation direction.
    """
    if (
        config.vehicle.waypoints_ned.ndim != 2
        or config.vehicle.waypoints_ned.shape[1] != 3
        or len(config.vehicle.waypoints_ned) < 2
    ):
        raise ValueError(
            "Vehicle waypoints must have shape (N, 3) with at least two "
            "waypoints to align vehicle approach."
        )

    target_speed = float(np.linalg.norm(target_velocity_ned))

    if target_speed <= SMALL_NUMBER:
        raise ValueError(
            "Target velocity must be nonzero to align vehicle approach."
        )

    vehicle_final_waypoint_ned = config.vehicle.waypoints_ned[-1]
    vehicle_penultimate_waypoint_ned = config.vehicle.waypoints_ned[-2]
    final_leg_vector_ned = (
        vehicle_final_waypoint_ned - vehicle_penultimate_waypoint_ned
    )
    final_leg_length = float(np.linalg.norm(final_leg_vector_ned))

    if final_leg_length <= SMALL_NUMBER:
        raise ValueError(
            "Vehicle final and next-to-last waypoints must be distinct "
            "to align vehicle approach."
        )

    target_direction_ned = target_velocity_ned / target_speed

    config.vehicle.waypoints_ned[-2] = (
        vehicle_final_waypoint_ned + target_direction_ned * final_leg_length
    )


def compute_time_to_minimum_distance_from_point(
    initial_position_ned: Vector3,
    velocity_ned: Vector3,
    point_ned: Vector3,
) -> float:
    """Return the future time at closest approach to a fixed point."""
    speed_squared = float(np.dot(velocity_ned, velocity_ned))

    if speed_squared <= SMALL_NUMBER:
        raise ValueError("Velocity must be nonzero for closest-approach time.")

    time_to_minimum_distance = float(
        np.dot(point_ned - initial_position_ned, velocity_ned)
        / speed_squared
    )

    return max(0.0, time_to_minimum_distance)


def compute_prelaunch_target_propagation(
    target_position_ned: Vector3,
    target_velocity_ned: Vector3,
    start_time: float,
    propagation_duration: float,
) -> PrelaunchTargetPropagation:
    """Return a 10 Hz target propagation to the predicted intercept point."""
    sample_time_step = 1.0 / PRELAUNCH_TARGET_PROPAGATION_RATE_HZ
    sample_count = int(math.ceil(propagation_duration / sample_time_step)) + 1
    elapsed_time_history = np.arange(sample_count, dtype=np.float64)
    elapsed_time_history *= sample_time_step
    elapsed_time_history = np.minimum(
        elapsed_time_history,
        propagation_duration,
    )
    time_history = start_time + elapsed_time_history
    target_position_history_ned = (
        target_position_ned
        + elapsed_time_history[:, np.newaxis] * target_velocity_ned
    )
    target_velocity_history_ned = np.repeat(
        target_velocity_ned[np.newaxis, :],
        sample_count,
        axis=0,
    )

    return PrelaunchTargetPropagation(
        time_history=time_history,
        target_position_history_ned=target_position_history_ned,
        target_velocity_history_ned=target_velocity_history_ned,
    )


def compute_path_time_to_final_waypoint(config: GuidanceConfig) -> float:
    """Return travel time from initial position through all waypoints."""
    path_points = np.vstack((config.initial_position_ned, config.waypoints_ned))
    segment_vectors = np.diff(path_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_path_length = float(np.sum(segment_lengths))

    return total_path_length / config.desired_flight_speed


def compute_synchronized_vehicle_launch_timing_from_target_state(
    config: SimulationConfig,
    target_position_ned: Vector3,
    target_velocity_ned: Vector3,
    current_time: float,
) -> tuple[float, float, float]:
    """Return target POCA, vehicle path time, and launch time."""
    vehicle_final_waypoint_ned = config.vehicle.waypoints_ned[-1]
    target_time_to_vehicle_final_waypoint_poca = (
        compute_time_to_minimum_distance_from_point(
            initial_position_ned=target_position_ned,
            velocity_ned=target_velocity_ned,
            point_ned=vehicle_final_waypoint_ned,
        )
    )
    vehicle_time_to_final_waypoint = compute_path_time_to_final_waypoint(
        config.vehicle
    )
    vehicle_launch_time = (
        current_time
        + target_time_to_vehicle_final_waypoint_poca
        - vehicle_time_to_final_waypoint
    )

    return (
        target_time_to_vehicle_final_waypoint_poca,
        vehicle_time_to_final_waypoint,
        vehicle_launch_time,
    )


def compute_synchronized_vehicle_launch_time_from_target_state(
    config: SimulationConfig,
    target_position_ned: Vector3,
    target_velocity_ned: Vector3,
    current_time: float,
) -> float:
    """Return launch time from a target state and current simulation time."""
    (
        _target_time_to_vehicle_final_waypoint_poca,
        _vehicle_time_to_final_waypoint,
        vehicle_launch_time,
    ) = compute_synchronized_vehicle_launch_timing_from_target_state(
        config=config,
        target_position_ned=target_position_ned,
        target_velocity_ned=target_velocity_ned,
        current_time=current_time,
    )

    return vehicle_launch_time


def compute_synchronized_vehicle_launch_time(config: SimulationConfig) -> float:
    """Launch the vehicle so it reaches its final waypoint with the target."""
    target_velocity_ned = compute_constant_heading_velocity_ned(config.target)
    align_vehicle_penultimate_waypoint_with_target_approach(
        config,
        target_velocity_ned,
    )

    return compute_synchronized_vehicle_launch_time_from_target_state(
        config=config,
        target_position_ned=config.target.initial_position_ned,
        target_velocity_ned=target_velocity_ned,
        current_time=0.0,
    )


def _require_text(parent: ET.Element, tag: str) -> str:
    child = parent.find(tag)

    if child is None or child.text is None:
        raise ValueError(f"Missing XML element: {tag}")

    return child.text.strip()


def _optional_text(parent: ET.Element, tag: str, default: str) -> str:
    child = parent.find(tag)

    if child is None or child.text is None:
        return default

    return child.text.strip()


def _load_xml_guidance(
    entity_element: ET.Element,
    entity_name: str,
) -> GuidanceConfig:
    guidance = entity_element.find("guidance")
    waypoints_element = entity_element.find("waypoints_ned")
    initial_position_element = entity_element.find("initial_position_ned")

    if guidance is None:
        raise ValueError(f"Missing XML element: {entity_name}/guidance")

    if waypoints_element is None:
        raise ValueError(f"Missing XML element: {entity_name}/waypoints_ned")

    if initial_position_element is None:
        raise ValueError(
            f"Missing XML element: {entity_name}/initial_position_ned"
        )

    waypoint_rows: list[list[float]] = []

    for waypoint in waypoints_element.findall("waypoint"):
        waypoint_rows.append(
            [
                float(waypoint.attrib["north"]),
                float(waypoint.attrib["east"]),
                float(waypoint.attrib["down"]),
            ]
        )

    return GuidanceConfig(
        waypoints_ned=np.asarray(waypoint_rows, dtype=np.float64),
        initial_position_ned=np.asarray(
            [
                float(initial_position_element.attrib["north"]),
                float(initial_position_element.attrib["east"]),
                float(initial_position_element.attrib["down"]),
            ],
            dtype=np.float64,
        ),
        desired_flight_speed=float(
            _require_text(guidance, "desired_flight_speed_mps")
        ),
        max_waypoint_intercept_angle_rad=math.radians(
            float(_require_text(guidance, "max_waypoint_intercept_angle_deg"))
        ),
        convergence_time_constant=float(
            _require_text(guidance, "convergence_time_constant_s")
        ),
        waypoint_plane_tolerance=float(
            _require_text(guidance, "waypoint_plane_tolerance_m")
        ),
    )


def load_xml_config(input_path: Path) -> SimulationConfig:
    root = ET.parse(input_path).getroot()

    metadata = root.find("metadata")
    simulation = root.find("simulation")
    vehicle = root.find("vehicle")
    target = root.find("target")

    if metadata is None:
        raise ValueError("Missing XML element: metadata")

    if simulation is None:
        raise ValueError("Missing XML element: simulation")

    if vehicle is None:
        raise ValueError("Missing XML element: vehicle")

    if target is None:
        raise ValueError("Missing XML element: target")

    integrated_pn_element = vehicle.find(
        "integrated_proportional_navigation"
    )

    if integrated_pn_element is None:
        raise ValueError(
            "Missing XML element: "
            "vehicle/integrated_proportional_navigation"
        )


    config = SimulationConfig(
        version=_require_text(metadata, "version"),
        scenario_title=_require_text(metadata, "scenario_title"),
        classification=_require_text(metadata, "classification"),
        dataset_name=_require_text(metadata, "dataset_name"),
        vehicle_launch_time=float(
            _optional_text(vehicle, "launch_time_s", "0.0")
        ),
        vehicle=_load_xml_guidance(vehicle, "vehicle"),
        target=_load_xml_guidance(target, "target"),
        integrated_pn=IntegratedPNConfig(
            terminal_range=float(
                _require_text(integrated_pn_element, "terminal_range_m")
            ),
            horizontal_navigation_constant=float(
                _require_text(
                    integrated_pn_element,
                    "horizontal_navigation_constant",
                )
            ),
            vertical_navigation_constant=float(
                _require_text(
                    integrated_pn_element,
                    "vertical_navigation_constant",
                )
            ),
        ),
        time_step=float(_require_text(simulation, "time_step_s")),
        simulation_duration=float(_require_text(simulation, "duration_s")),
        output_file=Path(_require_text(simulation, "output_file")),
    )

    config.vehicle_launch_time = compute_synchronized_vehicle_launch_time(
        config
    )

    _validate_config(config)
    return config


def load_config(input_path: Path) -> SimulationConfig:
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        return load_json_config(input_path)

    if suffix == ".xml":
        return load_xml_config(input_path)

    raise ValueError(
        f"Unsupported input format '{input_path.suffix}'. Use .json or .xml."
    )


def compute_velocity_command_ned(
    waypoint_positions_ned: Sequence[Vector3],
    waypoint_index: int,
    last_waypoint_position_ned: Vector3,
    position_ned: Vector3,
    desired_flight_speed: float,
    max_waypoint_intercept_angle: float,
    convergence_time_constant: float,
) -> tuple[Vector3, float, float]:
    current_waypoint_position_ned = np.asarray(
        waypoint_positions_ned[waypoint_index],
        dtype=np.float64,
    )
    last_waypoint_position_ned = np.asarray(
        last_waypoint_position_ned,
        dtype=np.float64,
    )
    position_ned = np.asarray(position_ned, dtype=np.float64)

    velocity_command_ned = np.zeros(3, dtype=np.float64)
    path_vector_ned = (
        current_waypoint_position_ned - last_waypoint_position_ned
    )
    path_length = float(np.linalg.norm(path_vector_ned))

    if path_length <= SMALL_NUMBER:
        return velocity_command_ned, 0.0, path_length

    path_direction_ned = path_vector_ned / path_length

    along_track_distance = float(
        np.dot(
            position_ned - last_waypoint_position_ned,
            path_direction_ned,
        )
    )

    clamped_along_track_distance = float(
        np.clip(along_track_distance, 0.0, path_length)
    )

    closest_path_position_ned = (
        last_waypoint_position_ned
        + clamped_along_track_distance * path_direction_ned
    )

    path_error_ned = position_ned - closest_path_position_ned

    cross_track_error_ned = (
        path_error_ned
        - np.dot(path_error_ned, path_direction_ned) * path_direction_ned
    )

    cross_track_error = float(np.linalg.norm(cross_track_error_ned))
    cross_track_direction_ned = np.zeros(3, dtype=np.float64)

    if cross_track_error > SMALL_NUMBER:
        cross_track_direction_ned = cross_track_error_ned / cross_track_error

    convergence_distance = (
        convergence_time_constant * desired_flight_speed
    )

    intercept_angle = (
        max_waypoint_intercept_angle
        * math.tanh(cross_track_error / convergence_distance)
    )

    guidance_direction_ned = (
        math.cos(intercept_angle) * path_direction_ned
        - math.sin(intercept_angle) * cross_track_direction_ned
    )

    velocity_command_ned = desired_flight_speed * guidance_direction_ned

    return velocity_command_ned, along_track_distance, path_length


def compute_entity_guidance(
    config: GuidanceConfig,
    state: EntityState,
) -> EntityGuidanceResult:
    waypoint_index = state.waypoint_index

    if waypoint_index >= len(config.waypoints_ned):
        return EntityGuidanceResult(
            velocity_ned=state.velocity_ned,
            waypoint_index=waypoint_index,
        )

    last_waypoint_position_ned = config.waypoints_ned[
        waypoint_index - 1
    ]

    (
        velocity_command_ned,
        along_track_distance,
        path_length,
    ) = compute_velocity_command_ned(
        waypoint_positions_ned=config.waypoints_ned,
        waypoint_index=waypoint_index,
        last_waypoint_position_ned=last_waypoint_position_ned,
        position_ned=state.position_ned,
        desired_flight_speed=config.desired_flight_speed,
        max_waypoint_intercept_angle=(
            config.max_waypoint_intercept_angle_rad
        ),
        convergence_time_constant=config.convergence_time_constant,
    )

    waypoint_plane_crossed = (
        along_track_distance
        >= path_length - config.waypoint_plane_tolerance
    )

    if waypoint_plane_crossed:
        waypoint_index += 1

        if waypoint_index >= len(config.waypoints_ned):
            print("Passed final waypoint, holding velocity until terminal mode.")
            return EntityGuidanceResult(
                velocity_ned=state.velocity_ned,
                waypoint_index=waypoint_index,
            )

        last_waypoint_position_ned = config.waypoints_ned[
            waypoint_index - 1
        ]

        (
            velocity_command_ned,
            _along_track_distance,
            _path_length,
        ) = compute_velocity_command_ned(
            waypoint_positions_ned=config.waypoints_ned,
            waypoint_index=waypoint_index,
            last_waypoint_position_ned=last_waypoint_position_ned,
            position_ned=state.position_ned,
            desired_flight_speed=config.desired_flight_speed,
            max_waypoint_intercept_angle=(
                config.max_waypoint_intercept_angle_rad
            ),
            convergence_time_constant=config.convergence_time_constant,
        )

    return EntityGuidanceResult(
        velocity_ned=velocity_command_ned,
        waypoint_index=waypoint_index,
    )



def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi

    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi

    return wrapped


def los_angles_ned(
    vehicle_position_ned: Vector3,
    target_position_ned: Vector3,
) -> tuple[float, float]:
    """
    Return scalar LOS azimuth and elevation angles.

    Azimuth:
      zero = north, positive toward east.

    Elevation:
      zero = local horizontal, positive upward.
    """
    relative_position_ned = target_position_ned - vehicle_position_ned

    north = float(relative_position_ned[0])
    east = float(relative_position_ned[1])
    down = float(relative_position_ned[2])

    horizontal_range = math.hypot(north, east)
    azimuth_rad = math.atan2(east, north)
    elevation_rad = math.atan2(-down, horizontal_range)

    return azimuth_rad, elevation_rad


def velocity_angles_ned(
    velocity_ned: Vector3,
) -> tuple[float, float]:
    """
    Return scalar inertial heading and flight-path angle.

    Heading:
      zero = north, positive toward east.

    Flight-path angle:
      zero = horizontal, positive upward.
    """
    north_velocity = float(velocity_ned[0])
    east_velocity = float(velocity_ned[1])
    down_velocity = float(velocity_ned[2])

    horizontal_speed = math.hypot(north_velocity, east_velocity)
    heading_rad = math.atan2(east_velocity, north_velocity)
    flight_path_angle_rad = math.atan2(
        -down_velocity,
        horizontal_speed,
    )

    return heading_rad, flight_path_angle_rad


def compute_two_angle_integrated_pn_velocity_ned(
    previous_los_azimuth_rad: float,
    current_los_azimuth_rad: float,
    previous_los_elevation_rad: float,
    current_los_elevation_rad: float,
    previous_commanded_heading_rad: float,
    previous_commanded_flight_path_angle_rad: float,
    desired_flight_speed: float,
    horizontal_navigation_constant: float,
    vertical_navigation_constant: float,
) -> tuple[Vector3, float, float]:
    """
    Apply two independent scalar integrated-PN channels.

    Horizontal:
      chi[k] = chi[k-1] + N_h * wrap(az[k] - az[k-1])

    Vertical:
      gamma[k] = gamma[k-1] + N_v * wrap(el[k] - el[k-1])

    No 3-D LOS vector rotation, target velocity, closing velocity, range rate,
    or LOS-rate estimate is used.
    """
    delta_los_azimuth = wrap_to_pi(
        current_los_azimuth_rad - previous_los_azimuth_rad
    )
    delta_los_elevation = wrap_to_pi(
        current_los_elevation_rad - previous_los_elevation_rad
    )

    commanded_heading_rad = wrap_to_pi(
        previous_commanded_heading_rad
        + horizontal_navigation_constant * delta_los_azimuth
    )

    commanded_flight_path_angle_rad = wrap_to_pi(
        previous_commanded_flight_path_angle_rad
        + vertical_navigation_constant * delta_los_elevation
    )

    # Limit gamma to a physically unambiguous velocity decomposition.
    commanded_flight_path_angle_rad = float(
        np.clip(
            commanded_flight_path_angle_rad,
            -0.5 * math.pi,
            0.5 * math.pi,
        )
    )

    horizontal_speed = (
        desired_flight_speed
        * math.cos(commanded_flight_path_angle_rad)
    )

    velocity_command_ned = np.asarray(
        [
            horizontal_speed * math.cos(commanded_heading_rad),
            horizontal_speed * math.sin(commanded_heading_rad),
            -desired_flight_speed
            * math.sin(commanded_flight_path_angle_rad),
        ],
        dtype=np.float64,
    )

    return (
        velocity_command_ned,
        commanded_heading_rad,
        commanded_flight_path_angle_rad,
    )



def ned_to_enu(vector_ned: Vector3) -> Vector3:
    return np.asarray(
        [vector_ned[1], vector_ned[0], -vector_ned[2]],
        dtype=np.float64,
    )


def run_simulation(config: SimulationConfig) -> SimulationResult:
    sample_time_step = min(
        config.time_step,
        LAUNCH_RECOMPUTE_INTERVAL_S,
    )
    number_of_steps = int(
        math.ceil(config.simulation_duration / sample_time_step)
    )
    maximum_samples = number_of_steps + 2

    time_history = np.zeros(maximum_samples, dtype=np.float64)
    vehicle_position_history_ned = np.zeros(
        (maximum_samples, 3), dtype=np.float64
    )
    vehicle_velocity_history_ned = np.zeros(
        (maximum_samples, 3), dtype=np.float64
    )
    target_position_history_ned = np.zeros(
        (maximum_samples, 3), dtype=np.float64
    )
    target_velocity_history_ned = np.zeros(
        (maximum_samples, 3), dtype=np.float64
    )
    range_history = np.zeros(maximum_samples, dtype=np.float64)
    terminal_guidance_history = np.zeros(
        maximum_samples, dtype=np.float64
    )

    vehicle_state = EntityState(
        position_ned=config.vehicle.initial_position_ned.copy()
    )
    target_state = EntityState(
        position_ned=config.target.initial_position_ned.copy()
    )

    vehicle_position_history_ned[0] = vehicle_state.position_ned
    target_position_history_ned[0] = target_state.position_ned
    range_history[0] = np.linalg.norm(
        target_state.position_ned - vehicle_state.position_ned
    )

    valid_sample_count = 1
    poca_time = 0.0
    poca_range = float(range_history[0])
    terminal_guidance_activation_time: float | None = None
    next_launch_time_recompute = LAUNCH_RECOMPUTE_INTERVAL_S
    target_time_to_vehicle_final_waypoint_poca = (
        compute_time_to_minimum_distance_from_point(
            initial_position_ned=target_state.position_ned,
            velocity_ned=compute_constant_heading_velocity_ned(config.target),
            point_ned=config.vehicle.waypoints_ned[-1],
        )
    )
    vehicle_time_to_final_waypoint = compute_path_time_to_final_waypoint(
        config.vehicle
    )
    prelaunch_target_propagations: list[PrelaunchTargetPropagation] = []

    if 0.0 < config.vehicle_launch_time:
        prelaunch_target_propagations.append(
            compute_prelaunch_target_propagation(
                target_position_ned=target_state.position_ned,
                target_velocity_ned=compute_constant_heading_velocity_ned(
                    config.target
                ),
                start_time=0.0,
                propagation_duration=(
                    target_time_to_vehicle_final_waypoint_poca
                ),
            )
        )

    for _step in range(number_of_steps):
        current_time = time_history[valid_sample_count - 1]

        target_guidance = compute_entity_guidance(
            config.target,
            target_state,
        )
        target_state.waypoint_index = target_guidance.waypoint_index
        target_velocity_ned = target_guidance.velocity_ned

        if current_time < config.vehicle_launch_time:
            vehicle_velocity_ned = np.zeros(3, dtype=np.float64)
            vehicle_state.velocity_ned = vehicle_velocity_ned.copy()
            integration_time = min(
                config.time_step,
                config.vehicle_launch_time - current_time,
                next_launch_time_recompute - current_time,
            )

            target_state.position_ned = (
                target_state.position_ned
                + target_velocity_ned * integration_time
            )

            sample_index = valid_sample_count
            time_history[sample_index] = current_time + integration_time
            vehicle_position_history_ned[sample_index] = (
                vehicle_state.position_ned
            )
            vehicle_velocity_history_ned[sample_index] = vehicle_velocity_ned
            target_position_history_ned[sample_index] = (
                target_state.position_ned
            )
            target_velocity_history_ned[sample_index] = target_velocity_ned

            relative_position_at_sample = (
                target_state.position_ned - vehicle_state.position_ned
            )
            range_history[sample_index] = np.linalg.norm(
                relative_position_at_sample
            )
            terminal_guidance_history[sample_index] = 0.0

            valid_sample_count += 1

            poca_time = float(time_history[sample_index])
            poca_range = float(range_history[sample_index])

            if (
                poca_time >= next_launch_time_recompute
                and poca_time < config.vehicle_launch_time
            ):
                align_vehicle_penultimate_waypoint_with_target_approach(
                    config,
                    target_velocity_ned,
                )
                (
                    target_time_to_vehicle_final_waypoint_poca,
                    vehicle_time_to_final_waypoint,
                    config.vehicle_launch_time,
                ) = (
                    compute_synchronized_vehicle_launch_timing_from_target_state(
                        config=config,
                        target_position_ned=target_state.position_ned,
                        target_velocity_ned=target_velocity_ned,
                        current_time=poca_time,
                    )
                )
                prelaunch_target_propagations.append(
                    compute_prelaunch_target_propagation(
                        target_position_ned=target_state.position_ned,
                        target_velocity_ned=target_velocity_ned,
                        start_time=poca_time,
                        propagation_duration=(
                            target_time_to_vehicle_final_waypoint_poca
                        ),
                    )
                )
                next_launch_time_recompute = (
                    poca_time + LAUNCH_RECOMPUTE_INTERVAL_S
                )

            if poca_time >= config.simulation_duration:
                break

            continue

        relative_position_ned = (
            target_state.position_ned - vehicle_state.position_ned
        )
        current_range = float(np.linalg.norm(relative_position_ned))

        if (
            not vehicle_state.terminal_guidance_active
            and current_range <= config.integrated_pn.terminal_range
        ):
            vehicle_state.terminal_guidance_active = True
            terminal_guidance_activation_time = current_time

            vehicle_guidance = compute_entity_guidance(
                config.vehicle,
                vehicle_state,
            )
            vehicle_state.waypoint_index = (
                vehicle_guidance.waypoint_index
            )
            vehicle_state.velocity_ned = (
                vehicle_guidance.velocity_ned.copy()
            )
            (
                vehicle_state.previous_los_azimuth_rad,
                vehicle_state.previous_los_elevation_rad,
            ) = los_angles_ned(
                vehicle_state.position_ned,
                target_state.position_ned,
            )

            (
                vehicle_state.commanded_heading_rad,
                vehicle_state.commanded_flight_path_angle_rad,
            ) = velocity_angles_ned(vehicle_state.velocity_ned)

        if vehicle_state.terminal_guidance_active:
            (
                current_los_azimuth_rad,
                current_los_elevation_rad,
            ) = los_angles_ned(
                vehicle_state.position_ned,
                target_state.position_ned,
            )

            if vehicle_state.velocity_ned is None:
                horizontal_speed = (
                    config.vehicle.desired_flight_speed
                    * math.cos(current_los_elevation_rad)
                )
                vehicle_state.velocity_ned = np.asarray(
                    [
                        horizontal_speed
                        * math.cos(current_los_azimuth_rad),
                        horizontal_speed
                        * math.sin(current_los_azimuth_rad),
                        -config.vehicle.desired_flight_speed
                        * math.sin(current_los_elevation_rad),
                    ],
                    dtype=np.float64,
                )

            if vehicle_state.previous_los_azimuth_rad is None:
                vehicle_state.previous_los_azimuth_rad = (
                    current_los_azimuth_rad
                )

            if vehicle_state.previous_los_elevation_rad is None:
                vehicle_state.previous_los_elevation_rad = (
                    current_los_elevation_rad
                )

            if (
                vehicle_state.commanded_heading_rad is None
                or vehicle_state.commanded_flight_path_angle_rad is None
            ):
                (
                    vehicle_state.commanded_heading_rad,
                    vehicle_state.commanded_flight_path_angle_rad,
                ) = velocity_angles_ned(vehicle_state.velocity_ned)

            (
                vehicle_velocity_ned,
                vehicle_state.commanded_heading_rad,
                vehicle_state.commanded_flight_path_angle_rad,
            ) = compute_two_angle_integrated_pn_velocity_ned(
                previous_los_azimuth_rad=(
                    vehicle_state.previous_los_azimuth_rad
                ),
                current_los_azimuth_rad=current_los_azimuth_rad,
                previous_los_elevation_rad=(
                    vehicle_state.previous_los_elevation_rad
                ),
                current_los_elevation_rad=current_los_elevation_rad,
                previous_commanded_heading_rad=(
                    vehicle_state.commanded_heading_rad
                ),
                previous_commanded_flight_path_angle_rad=(
                    vehicle_state.commanded_flight_path_angle_rad
                ),
                desired_flight_speed=(
                    config.vehicle.desired_flight_speed
                ),
                horizontal_navigation_constant=(
                    config.integrated_pn.horizontal_navigation_constant
                ),
                vertical_navigation_constant=(
                    config.integrated_pn.vertical_navigation_constant
                ),
            )

            vehicle_state.velocity_ned = vehicle_velocity_ned.copy()
            vehicle_state.previous_los_azimuth_rad = (
                current_los_azimuth_rad
            )
            vehicle_state.previous_los_elevation_rad = (
                current_los_elevation_rad
            )
        else:
            vehicle_guidance = compute_entity_guidance(
                config.vehicle,
                vehicle_state,
            )
            vehicle_state.waypoint_index = (
                vehicle_guidance.waypoint_index
            )
            vehicle_velocity_ned = vehicle_guidance.velocity_ned
            vehicle_state.velocity_ned = vehicle_velocity_ned.copy()

        relative_position_ned = (
            target_state.position_ned - vehicle_state.position_ned
        )
        relative_velocity_ned = (
            target_velocity_ned - vehicle_velocity_ned
        )

        relative_speed_squared = float(
            np.dot(relative_velocity_ned, relative_velocity_ned)
        )

        if relative_speed_squared > SMALL_NUMBER:
            time_to_step_poca = -float(
                np.dot(relative_position_ned, relative_velocity_ned)
            ) / relative_speed_squared
        else:
            time_to_step_poca = math.inf

        poca_occurs_this_step = (
            0.0 <= time_to_step_poca <= config.time_step
        )

        if poca_occurs_this_step:
            integration_time = time_to_step_poca
        else:
            integration_time = config.time_step

        vehicle_state.position_ned = (
            vehicle_state.position_ned
            + vehicle_velocity_ned * integration_time
        )
        target_state.position_ned = (
            target_state.position_ned
            + target_velocity_ned * integration_time
        )

        sample_index = valid_sample_count
        time_history[sample_index] = current_time + integration_time
        vehicle_position_history_ned[sample_index] = (
            vehicle_state.position_ned
        )
        vehicle_velocity_history_ned[sample_index] = vehicle_velocity_ned
        target_position_history_ned[sample_index] = target_state.position_ned
        target_velocity_history_ned[sample_index] = target_velocity_ned

        relative_position_at_sample = (
            target_state.position_ned - vehicle_state.position_ned
        )
        range_history[sample_index] = np.linalg.norm(
            relative_position_at_sample
        )
        terminal_guidance_history[sample_index] = (
            1.0 if vehicle_state.terminal_guidance_active else 0.0
        )

        valid_sample_count += 1

        if poca_occurs_this_step:
            poca_time = float(time_history[sample_index])
            poca_range = float(range_history[sample_index])
            break

        poca_time = float(time_history[sample_index])
        poca_range = float(range_history[sample_index])

        if poca_time >= config.simulation_duration:
            break

    return SimulationResult(
        time_history=time_history[:valid_sample_count],
        vehicle_position_history_ned=(
            vehicle_position_history_ned[:valid_sample_count]
        ),
        vehicle_velocity_history_ned=(
            vehicle_velocity_history_ned[:valid_sample_count]
        ),
        target_position_history_ned=(
            target_position_history_ned[:valid_sample_count]
        ),
        target_velocity_history_ned=(
            target_velocity_history_ned[:valid_sample_count]
        ),
        range_history=range_history[:valid_sample_count],
        terminal_guidance_history=(
            terminal_guidance_history[:valid_sample_count]
        ),
        poca_time=poca_time,
        poca_range=poca_range,
        terminal_guidance_activation_time=(
            terminal_guidance_activation_time
        ),
        target_time_to_vehicle_final_waypoint_poca=(
            target_time_to_vehicle_final_waypoint_poca
        ),
        vehicle_time_to_final_waypoint=vehicle_time_to_final_waypoint,
        vehicle_launch_time=config.vehicle_launch_time,
        prelaunch_target_propagations=prelaunch_target_propagations,
    )


def write_translational_state_out(
    output_path: Path,
    config: SimulationConfig,
    result: SimulationResult,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variable_names = [
        "time",
        "vehiclePositionEast",
        "vehiclePositionNorth",
        "vehiclePositionUp",
        "vehicleVelocityEast",
        "vehicleVelocityNorth",
        "vehicleVelocityUp",
        "targetPositionEast",
        "targetPositionNorth",
        "targetPositionUp",
        "targetVelocityEast",
        "targetVelocityNorth",
        "targetVelocityUp",
        "vehicleToTargetRange",
        "integratedPNActive",
    ]

    variable_units = [
        "s",
        "m",
        "m",
        "m",
        "m/s",
        "m/s",
        "m/s",
        "m",
        "m",
        "m",
        "m/s",
        "m/s",
        "m/s",
        "m",
        "nd",
    ]

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(
            "1 Total Datasets "
            f"BetaFlight INS V{config.version} "
            f"Scenario Title: {config.scenario_title} "
            f"Classification: {config.classification}\n"
        )
        output_file.write(f"dataset: {config.dataset_name}\n")
        output_file.write(f"{len(variable_names)}\n")
        output_file.write("\t".join(variable_names) + "\n")
        output_file.write("\t".join(variable_units) + "\n")

        for sample_index in range(len(result.time_history)):
            vehicle_position_enu = ned_to_enu(
                result.vehicle_position_history_ned[sample_index]
            )
            vehicle_velocity_enu = ned_to_enu(
                result.vehicle_velocity_history_ned[sample_index]
            )
            target_position_enu = ned_to_enu(
                result.target_position_history_ned[sample_index]
            )
            target_velocity_enu = ned_to_enu(
                result.target_velocity_history_ned[sample_index]
            )

            row = [
                result.time_history[sample_index],
                vehicle_position_enu[0],
                vehicle_position_enu[1],
                vehicle_position_enu[2],
                vehicle_velocity_enu[0],
                vehicle_velocity_enu[1],
                vehicle_velocity_enu[2],
                target_position_enu[0],
                target_position_enu[1],
                target_position_enu[2],
                target_velocity_enu[0],
                target_velocity_enu[1],
                target_velocity_enu[2],
                result.range_history[sample_index],
                result.terminal_guidance_history[sample_index],
            ]

            output_file.write(
                "\t".join(f"{value:.9f}" for value in row) + "\n"
            )


def write_prelaunch_target_propagations_out(
    output_path: Path,
    config: SimulationConfig,
    result: SimulationResult,
) -> None:
    """Write all pre-launch target propagations at 10 Hz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variable_names = [
        "targetPositionEast",
        "targetPositionNorth",
        "targetPositionUp",
        "targetVelocityEast",
        "targetVelocityNorth",
        "targetVelocityUp",
    ]

    variable_units = [
        "m",
        "m",
        "m",
        "m/s",
        "m/s",
        "m/s",
    ]

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(
            f"{len(result.prelaunch_target_propagations)} Total Datasets "
            f"BetaFlight INS V{config.version} "
            f"Scenario Title: {config.scenario_title} "
            f"Classification: {config.classification}\n"
        )

        for propagation_index, propagation in enumerate(
            result.prelaunch_target_propagations,
            start=1,
        ):
            output_file.write(
                f"dataset: {config.dataset_name} "
                f"prelaunchTargetPropagation{propagation_index}\n"
            )
            output_file.write(f"{len(variable_names)}\n")
            output_file.write("\t".join(variable_names) + "\n")
            output_file.write("\t".join(variable_units) + "\n")

            for sample_index in range(len(propagation.time_history)):
                target_position_enu = ned_to_enu(
                    propagation.target_position_history_ned[sample_index]
                )
                target_velocity_enu = ned_to_enu(
                    propagation.target_velocity_history_ned[sample_index]
                )

                row = [
                    target_position_enu[0],
                    target_position_enu[1],
                    target_position_enu[2],
                    target_velocity_enu[0],
                    target_velocity_enu[1],
                    target_velocity_enu[2],
                ]

                output_file.write(
                    "\t".join(f"{value:.9f}" for value in row) + "\n"
                )


def resolve_output_path(
    config: SimulationConfig,
    input_path: Path,
    command_line_output: Path | None,
) -> Path:
    if command_line_output is not None:
        return command_line_output

    if config.output_file.is_absolute():
        return config.output_file

    return input_path.parent / config.output_file


def resolve_prelaunch_target_propagations_output_path(
    output_path: Path,
) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_prelaunch_target_propagations"
        f"{output_path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run VFG trajectories with two-angle integrated PN, stop at POCA, "
            "and write ENU translational state data."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="JSON or XML simulation input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .out path overriding the input-file setting.",
    )

    args = parser.parse_args()
    config = load_config(args.input_file)
    result = run_simulation(config)

    output_path = resolve_output_path(
        config,
        args.input_file,
        args.output,
    )

    write_translational_state_out(
        output_path=output_path,
        config=config,
        result=result,
    )
    prelaunch_target_propagations_output_path = (
        resolve_prelaunch_target_propagations_output_path(output_path)
    )
    write_prelaunch_target_propagations_out(
        output_path=prelaunch_target_propagations_output_path,
        config=config,
        result=result,
    )

    print(
        "Target time to minimum distance from vehicle final waypoint: "
        f"{result.target_time_to_vehicle_final_waypoint_poca:.9f} s"
    )
    print(
        "Vehicle time through all waypoints to final waypoint: "
        f"{result.vehicle_time_to_final_waypoint:.9f} s"
    )
    print(
        "Computed synchronized vehicle launch time: "
        f"{result.vehicle_launch_time:.9f} s"
    )

    if result.terminal_guidance_activation_time is None:
        print("Integrated PN did not activate.")
    else:
        print(
            "Integrated PN activation time: "
            f"{result.terminal_guidance_activation_time:.9f} s"
        )

    print(f"POCA time: {result.poca_time:.9f} s")
    print(f"Vehicle-to-target range at POCA: {result.poca_range:.9f} m")
    print(f"Wrote {len(result.time_history)} samples to {output_path}")
    print(
        "Wrote "
        f"{len(result.prelaunch_target_propagations)} "
        "prelaunch target propagations to "
        f"{prelaunch_target_propagations_output_path}"
    )


if __name__ == "__main__":
    main()
