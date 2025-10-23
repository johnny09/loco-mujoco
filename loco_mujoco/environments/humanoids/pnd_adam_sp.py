from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class PndAdamSp(BaseRobotHumanoid):
    """
    Description
    ------------

    Mujoco environment of the PND Adam SP robot (with full hand control - 3 DOF wrists).

    Default Observation Space
    -----------------
    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 4        q_root                        FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_left_hip_pitch              JointPos         [-2.164]                             [2.164]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_left_hip_roll               JointPos         [-0.733]                             [1.605]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_left_hip_yaw                JointPos         [-0.785]                             [0.785]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_left_knee                   JointPos         [0.0]                                [2.391]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_left_ankle_pitch            JointPos         [-1.0]                               [0.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_left_ankle_roll             JointPos         [-0.3491]                            [0.3491]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_right_hip_pitch             JointPos         [-2.164]                             [2.164]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_right_hip_roll              JointPos         [-1.605]                             [0.733]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_right_hip_yaw               JointPos         [-0.785]                             [0.785]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_right_knee                  JointPos         [0.0]                                [2.391]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_right_ankle_pitch           JointPos         [-1.0]                               [0.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_right_ankle_roll            JointPos         [-0.3491]                            [0.3491]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_waist_roll                  JointPos         [-0.279]                             [0.279]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_waist_pitch                 JointPos         [-0.663]                             [1.361]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_waist_yaw                   JointPos         [-0.829]                             [0.829]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_left_shoulder_pitch         JointPos         [-3.613]                             [2.042]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_left_shoulder_roll          JointPos         [-0.628]                             [2.793]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_left_shoulder_yaw           JointPos         [-2.583]                             [2.583]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_left_elbow                  JointPos         [-2.496]                             [0.209]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_left_wrist_yaw              JointPos         [-2.67]                              [2.67]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_left_wrist_pitch            JointPos         [-0.96]                              [0.96]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_left_wrist_roll             JointPos         [-0.96]                              [0.96]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_right_shoulder_pitch        JointPos         [-3.613]                             [2.042]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_right_shoulder_roll         JointPos         [-2.793]                             [0.628]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_right_shoulder_yaw          JointPos         [-2.583]                             [2.583]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30           q_right_elbow                 JointPos         [-2.496]                             [0.209]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    31           q_right_wrist_yaw             JointPos         [-2.67]                              [2.67]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    32           q_right_wrist_pitch           JointPos         [-0.96]                              [0.96]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    33           q_right_wrist_roll            JointPos         [-0.96]                              [0.96]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    34 - 39      dq_root                       FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_left_hip_pitch             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_left_hip_roll              JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_left_hip_yaw               JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_left_knee                  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_left_ankle_pitch           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_left_ankle_roll            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_right_hip_pitch            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_right_hip_roll             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_right_hip_yaw              JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_right_knee                 JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_right_ankle_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_right_ankle_roll           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_waist_roll                 JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_waist_pitch                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_waist_yaw                  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_left_shoulder_pitch        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_left_shoulder_roll         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_left_shoulder_yaw          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_left_elbow                 JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_left_wrist_yaw             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_left_wrist_pitch           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    61           dq_left_wrist_roll            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    62           dq_right_shoulder_pitch       JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    63           dq_right_shoulder_roll        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    64           dq_right_shoulder_yaw         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    65           dq_right_elbow                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    66           dq_right_wrist_yaw            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    67           dq_right_wrist_pitch          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    68           dq_right_wrist_roll           JointVel         [-inf]                               [inf]                          1
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    ----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0 - 27          -1.0 1.0
    =============== ==== ===

    Methods
    ------------

    """

    mjx_enabled = False

    def __init__(
        self,
        spec: Union[str, MjSpec] = None,
        observation_spec: List[Observation] = None,
        actuation_spec: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            **kwargs,
        )

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: A list of observations.
        """

        observation_spec = [
            # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            # Left leg
            ObservationType.JointPos(
                "q_left_hip_pitch", xml_name="left_hip_pitch_joint"
            ),
            ObservationType.JointPos("q_left_hip_roll", xml_name="left_hip_roll_joint"),
            ObservationType.JointPos("q_left_hip_yaw", xml_name="left_hip_yaw_joint"),
            ObservationType.JointPos("q_left_knee", xml_name="left_knee_joint"),
            ObservationType.JointPos(
                "q_left_ankle_pitch", xml_name="left_ankle_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_ankle_roll", xml_name="left_ankle_roll_joint"
            ),
            # Right leg
            ObservationType.JointPos(
                "q_right_hip_pitch", xml_name="right_hip_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_hip_roll", xml_name="right_hip_roll_joint"
            ),
            ObservationType.JointPos("q_right_hip_yaw", xml_name="right_hip_yaw_joint"),
            ObservationType.JointPos("q_right_knee", xml_name="right_knee_joint"),
            ObservationType.JointPos(
                "q_right_ankle_pitch", xml_name="right_ankle_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_ankle_roll", xml_name="right_ankle_roll_joint"
            ),
            # Waist
            ObservationType.JointPos("q_waist_roll", xml_name="waist_roll_joint"),
            ObservationType.JointPos("q_waist_pitch", xml_name="waist_pitch_joint"),
            ObservationType.JointPos("q_waist_yaw", xml_name="waist_yaw_joint"),
            # Left arm
            ObservationType.JointPos(
                "q_left_shoulder_pitch", xml_name="left_shoulder_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_shoulder_roll", xml_name="left_shoulder_roll_joint"
            ),
            ObservationType.JointPos(
                "q_left_shoulder_yaw", xml_name="left_shoulder_yaw_joint"
            ),
            ObservationType.JointPos("q_left_elbow", xml_name="left_elbow_joint"),
            ObservationType.JointPos(
                "q_left_wrist_yaw", xml_name="left_wrist_yaw_joint"
            ),
            ObservationType.JointPos(
                "q_left_wrist_pitch", xml_name="left_wrist_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_left_wrist_roll", xml_name="left_wrist_roll_joint"
            ),
            # Right arm
            ObservationType.JointPos(
                "q_right_shoulder_pitch", xml_name="right_shoulder_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_shoulder_roll", xml_name="right_shoulder_roll_joint"
            ),
            ObservationType.JointPos(
                "q_right_shoulder_yaw", xml_name="right_shoulder_yaw_joint"
            ),
            ObservationType.JointPos("q_right_elbow", xml_name="right_elbow_joint"),
            ObservationType.JointPos(
                "q_right_wrist_yaw", xml_name="right_wrist_yaw_joint"
            ),
            ObservationType.JointPos(
                "q_right_wrist_pitch", xml_name="right_wrist_pitch_joint"
            ),
            ObservationType.JointPos(
                "q_right_wrist_roll", xml_name="right_wrist_roll_joint"
            ),
            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            # Left leg
            ObservationType.JointVel(
                "dq_left_hip_pitch", xml_name="left_hip_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_hip_roll", xml_name="left_hip_roll_joint"
            ),
            ObservationType.JointVel("dq_left_hip_yaw", xml_name="left_hip_yaw_joint"),
            ObservationType.JointVel("dq_left_knee", xml_name="left_knee_joint"),
            ObservationType.JointVel(
                "dq_left_ankle_pitch", xml_name="left_ankle_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_ankle_roll", xml_name="left_ankle_roll_joint"
            ),
            # Right leg
            ObservationType.JointVel(
                "dq_right_hip_pitch", xml_name="right_hip_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_hip_roll", xml_name="right_hip_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_hip_yaw", xml_name="right_hip_yaw_joint"
            ),
            ObservationType.JointVel("dq_right_knee", xml_name="right_knee_joint"),
            ObservationType.JointVel(
                "dq_right_ankle_pitch", xml_name="right_ankle_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_ankle_roll", xml_name="right_ankle_roll_joint"
            ),
            # Waist
            ObservationType.JointVel("dq_waist_roll", xml_name="waist_roll_joint"),
            ObservationType.JointVel("dq_waist_pitch", xml_name="waist_pitch_joint"),
            ObservationType.JointVel("dq_waist_yaw", xml_name="waist_yaw_joint"),
            # Left arm
            ObservationType.JointVel(
                "dq_left_shoulder_pitch", xml_name="left_shoulder_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_shoulder_roll", xml_name="left_shoulder_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_left_shoulder_yaw", xml_name="left_shoulder_yaw_joint"
            ),
            ObservationType.JointVel("dq_left_elbow", xml_name="left_elbow_joint"),
            ObservationType.JointVel(
                "dq_left_wrist_yaw", xml_name="left_wrist_yaw_joint"
            ),
            ObservationType.JointVel(
                "dq_left_wrist_pitch", xml_name="left_wrist_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_left_wrist_roll", xml_name="left_wrist_roll_joint"
            ),
            # Right arm
            ObservationType.JointVel(
                "dq_right_shoulder_pitch", xml_name="right_shoulder_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_shoulder_roll", xml_name="right_shoulder_roll_joint"
            ),
            ObservationType.JointVel(
                "dq_right_shoulder_yaw", xml_name="right_shoulder_yaw_joint"
            ),
            ObservationType.JointVel("dq_right_elbow", xml_name="right_elbow_joint"),
            ObservationType.JointVel(
                "dq_right_wrist_yaw", xml_name="right_wrist_yaw_joint"
            ),
            ObservationType.JointVel(
                "dq_right_wrist_pitch", xml_name="right_wrist_pitch_joint"
            ),
            ObservationType.JointVel(
                "dq_right_wrist_roll", xml_name="right_wrist_roll_joint"
            ),
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Returns the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: A list of actuator names.
        """
        return [actuator.name for actuator in spec.actuators]

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the PND Adam SP environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "pnd_adam_sp" / "adam_sp.xml").as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.
        """
        return "torso_link"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint in the Mujoco XML file.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height.

        Returns:
            Tuple[float, float]: The healthy height range (min, max).
        """
        return (0.6, 1.3)
