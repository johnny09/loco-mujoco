from typing import Tuple, List, Union
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class PndAdamLite(BaseRobotHumanoid):
    """
    Description
    ------------

    Mujoco environment of the PND Adam Lite robot.


    Default Observation Space
    -----------------
    ============ ============================= ================ ==================================== ============================== ===
    Index in Obs Name                          ObservationType  Min                                  Max                            Dim
    ============ ============================= ================ ==================================== ============================== ===
    0 - 4        q_root                        FreeJointPosNoXY [-inf, -inf, -inf, -inf, -inf]       [inf, inf, inf, inf, inf]      5
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    5            q_hipPitch_Left               JointPos         [-2.164]                             [2.164]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    6            q_hipRoll_Left                JointPos         [-0.733]                             [1.605]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    7            q_hipYaw_Left                 JointPos         [-0.785]                             [0.785]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    8            q_kneePitch_Left              JointPos         [0.052]                              [2.391]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    9            q_anklePitch_Left             JointPos         [-1.0]                               [0.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    10           q_ankleRoll_Left              JointPos         [-0.3491]                            [0.3491]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    11           q_hipPitch_Right              JointPos         [-2.164]                             [2.164]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    12           q_hipRoll_Right               JointPos         [-1.605]                             [0.733]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    13           q_hipYaw_Right                JointPos         [-0.785]                             [0.785]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    14           q_kneePitch_Right             JointPos         [0.052]                              [2.391]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    15           q_anklePitch_Right            JointPos         [-1.0]                               [0.35]                         1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    16           q_ankleRoll_Right             JointPos         [-0.3491]                            [0.3491]                       1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    17           q_waistRoll                   JointPos         [-0.279]                             [0.279]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    18           q_waistPitch                  JointPos         [-0.663]                             [1.361]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    19           q_waistYaw                    JointPos         [-0.829]                             [0.829]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    20           q_shoulderPitch_Left          JointPos         [-3.613]                             [2.042]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    21           q_shoulderRoll_Left           JointPos         [-0.628]                             [2.793]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    22           q_shoulderYaw_Left            JointPos         [-2.583]                             [2.583]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    23           q_elbow_Left                  JointPos         [-2.496]                             [0.209]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    24           q_wristYaw_Left               JointPos         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    25           q_shoulderPitch_Right         JointPos         [-3.613]                             [2.042]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    26           q_shoulderRoll_Right          JointPos         [-2.793]                             [0.628]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    27           q_shoulderYaw_Right           JointPos         [-2.583]                             [2.583]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    28           q_elbow_Right                 JointPos         [-2.496]                             [0.209]                        1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    29           q_wristYaw_Right              JointPos         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    30 - 35      dq_root                       FreeJointVel     [-inf, -inf, -inf, -inf, -inf, -inf] [inf, inf, inf, inf, inf, inf] 6
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    36           dq_hipPitch_Left              JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    37           dq_hipRoll_Left               JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    38           dq_hipYaw_Left                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    39           dq_kneePitch_Left             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    40           dq_anklePitch_Left            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    41           dq_ankleRoll_Left             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    42           dq_hipPitch_Right             JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    43           dq_hipRoll_Right              JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    44           dq_hipYaw_Right               JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    45           dq_kneePitch_Right            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    46           dq_anklePitch_Right           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    47           dq_ankleRoll_Right            JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    48           dq_waistRoll                  JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    49           dq_waistPitch                 JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    50           dq_waistYaw                   JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    51           dq_shoulderPitch_Left         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    52           dq_shoulderRoll_Left          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    53           dq_shoulderYaw_Left           JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    54           dq_elbow_Left                 JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    55           dq_wristYaw_Left              JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    56           dq_shoulderPitch_Right        JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    57           dq_shoulderRoll_Right         JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    58           dq_shoulderYaw_Right          JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    59           dq_elbow_Right                JointVel         [-inf]                               [inf]                          1
    ------------ ----------------------------- ---------------- ------------------------------------ ------------------------------ ---
    60           dq_wristYaw_Right             JointVel         [-inf]                               [inf]                          1
    ============ ============================= ================ ==================================== ============================== ===

    Default Action Space
    ----------------

    Control function type: **DefaultControl**

    See control function interface for more details.

    =============== ==== ===
    Index in Action Min  Max
    =============== ==== ===
    0               -1.0 1.0
    --------------- ---- ---
    1               -1.0 1.0
    --------------- ---- ---
    2               -1.0 1.0
    --------------- ---- ---
    3               -1.0 1.0
    --------------- ---- ---
    4               -1.0 1.0
    --------------- ---- ---
    5               -1.0 1.0
    --------------- ---- ---
    6               -1.0 1.0
    --------------- ---- ---
    7               -1.0 1.0
    --------------- ---- ---
    8               -1.0 1.0
    --------------- ---- ---
    9               -1.0 1.0
    --------------- ---- ---
    10              -1.0 1.0
    --------------- ---- ---
    11              -1.0 1.0
    --------------- ---- ---
    12              -1.0 1.0
    --------------- ---- ---
    13              -1.0 1.0
    --------------- ---- ---
    14              -1.0 1.0
    --------------- ---- ---
    15              -1.0 1.0
    --------------- ---- ---
    16              -1.0 1.0
    --------------- ---- ---
    17              -1.0 1.0
    --------------- ---- ---
    18              -1.0 1.0
    --------------- ---- ---
    19              -1.0 1.0
    --------------- ---- ---
    20              -1.0 1.0
    --------------- ---- ---
    21              -1.0 1.0
    --------------- ---- ---
    22              -1.0 1.0
    --------------- ---- ---
    23              -1.0 1.0
    --------------- ---- ---
    24              -1.0 1.0
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
            ObservationType.FreeJointPosNoXY("q_root", xml_name="floating_base"),
            # Left leg
            ObservationType.JointPos("q_hipPitch_Left", xml_name="hipPitch_Left"),
            ObservationType.JointPos("q_hipRoll_Left", xml_name="hipRoll_Left"),
            ObservationType.JointPos("q_hipYaw_Left", xml_name="hipYaw_Left"),
            ObservationType.JointPos("q_kneePitch_Left", xml_name="kneePitch_Left"),
            ObservationType.JointPos("q_anklePitch_Left", xml_name="anklePitch_Left"),
            ObservationType.JointPos("q_ankleRoll_Left", xml_name="ankleRoll_Left"),
            # Right leg
            ObservationType.JointPos("q_hipPitch_Right", xml_name="hipPitch_Right"),
            ObservationType.JointPos("q_hipRoll_Right", xml_name="hipRoll_Right"),
            ObservationType.JointPos("q_hipYaw_Right", xml_name="hipYaw_Right"),
            ObservationType.JointPos("q_kneePitch_Right", xml_name="kneePitch_Right"),
            ObservationType.JointPos("q_anklePitch_Right", xml_name="anklePitch_Right"),
            ObservationType.JointPos("q_ankleRoll_Right", xml_name="ankleRoll_Right"),
            # Waist
            ObservationType.JointPos("q_waistRoll", xml_name="waistRoll"),
            ObservationType.JointPos("q_waistPitch", xml_name="waistPitch"),
            ObservationType.JointPos("q_waistYaw", xml_name="waistYaw"),
            # Left arm
            ObservationType.JointPos(
                "q_shoulderPitch_Left", xml_name="shoulderPitch_Left"
            ),
            ObservationType.JointPos(
                "q_shoulderRoll_Left", xml_name="shoulderRoll_Left"
            ),
            ObservationType.JointPos("q_shoulderYaw_Left", xml_name="shoulderYaw_Left"),
            ObservationType.JointPos("q_elbow_Left", xml_name="elbow_Left"),
            ObservationType.JointPos("q_wristYaw_Left", xml_name="wristYaw_Left"),
            # Right arm
            ObservationType.JointPos(
                "q_shoulderPitch_Right", xml_name="shoulderPitch_Right"
            ),
            ObservationType.JointPos(
                "q_shoulderRoll_Right", xml_name="shoulderRoll_Right"
            ),
            ObservationType.JointPos(
                "q_shoulderYaw_Right", xml_name="shoulderYaw_Right"
            ),
            ObservationType.JointPos("q_elbow_Right", xml_name="elbow_Right"),
            ObservationType.JointPos("q_wristYaw_Right", xml_name="wristYaw_Right"),
            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="floating_base"),
            # Left leg
            ObservationType.JointVel("dq_hipPitch_Left", xml_name="hipPitch_Left"),
            ObservationType.JointVel("dq_hipRoll_Left", xml_name="hipRoll_Left"),
            ObservationType.JointVel("dq_hipYaw_Left", xml_name="hipYaw_Left"),
            ObservationType.JointVel("dq_kneePitch_Left", xml_name="kneePitch_Left"),
            ObservationType.JointVel("dq_anklePitch_Left", xml_name="anklePitch_Left"),
            ObservationType.JointVel("dq_ankleRoll_Left", xml_name="ankleRoll_Left"),
            # Right leg
            ObservationType.JointVel("dq_hipPitch_Right", xml_name="hipPitch_Right"),
            ObservationType.JointVel("dq_hipRoll_Right", xml_name="hipRoll_Right"),
            ObservationType.JointVel("dq_hipYaw_Right", xml_name="hipYaw_Right"),
            ObservationType.JointVel("dq_kneePitch_Right", xml_name="kneePitch_Right"),
            ObservationType.JointVel(
                "dq_anklePitch_Right", xml_name="anklePitch_Right"
            ),
            ObservationType.JointVel("dq_ankleRoll_Right", xml_name="ankleRoll_Right"),
            # Waist
            ObservationType.JointVel("dq_waistRoll", xml_name="waistRoll"),
            ObservationType.JointVel("dq_waistPitch", xml_name="waistPitch"),
            ObservationType.JointVel("dq_waistYaw", xml_name="waistYaw"),
            # Left arm
            ObservationType.JointVel(
                "dq_shoulderPitch_Left", xml_name="shoulderPitch_Left"
            ),
            ObservationType.JointVel(
                "dq_shoulderRoll_Left", xml_name="shoulderRoll_Left"
            ),
            ObservationType.JointVel(
                "dq_shoulderYaw_Left", xml_name="shoulderYaw_Left"
            ),
            ObservationType.JointVel("dq_elbow_Left", xml_name="elbow_Left"),
            ObservationType.JointVel("dq_wristYaw_Left", xml_name="wristYaw_Left"),
            # Right arm
            ObservationType.JointVel(
                "dq_shoulderPitch_Right", xml_name="shoulderPitch_Right"
            ),
            ObservationType.JointVel(
                "dq_shoulderRoll_Right", xml_name="shoulderRoll_Right"
            ),
            ObservationType.JointVel(
                "dq_shoulderYaw_Right", xml_name="shoulderYaw_Right"
            ),
            ObservationType.JointVel("dq_elbow_Right", xml_name="elbow_Right"),
            ObservationType.JointVel("dq_wristYaw_Right", xml_name="wristYaw_Right"),
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
        Returns the default XML file path for the PND Adam Lite environment.
        """
        return (
            loco_mujoco.PATH_TO_MODELS / "pnd_adam_lite" / "adam_lite.xml"
        ).as_posix()

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco XML file.
        """
        return "torso"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint in the Mujoco XML file.
        """
        return "floating_base"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height.

        Returns:
            Tuple[float, float]: The healthy height range (min, max).
        """
        return (0.6, 1.3)
