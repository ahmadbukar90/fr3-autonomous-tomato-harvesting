#include <cstdint>

#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <mujoco/mujoco.h>
#include <mujoco_ros2_control/mujoco_system_interface.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include <string>
#include <vector>

#include <cmath>

namespace mujoco_fr3_bringup
{

class FR3TomatoMujocoSystem
    : public mujoco_ros2_control::MujocoSystemInterface
{
public:
  FR3TomatoMujocoSystem() = default;

  ~FR3TomatoMujocoSystem() override
  {
    if (model_copy_ != nullptr)
    {
      mj_deleteModel(model_copy_);
      model_copy_ = nullptr;
    }
  }

  hardware_interface::CallbackReturn on_init(
      const hardware_interface::HardwareInfo & info) override
  {
    const auto result =
        mujoco_ros2_control::MujocoSystemInterface::on_init(
            info);

    if (
        result !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
      return result;
    }

    get_model(model_copy_);

    if (model_copy_ == nullptr)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Could not obtain MuJoCo model copy.");

      return hardware_interface::CallbackReturn::ERROR;
    }

    left_pad_geom_id_ = mj_name2id(
        model_copy_,
        mjOBJ_GEOM,
        "gelsight_left_collision");

    right_pad_geom_id_ = mj_name2id(
        model_copy_,
        mjOBJ_GEOM,
        "gelsight_right_collision");


    hand_body_id_ = mj_name2id(
        model_copy_,
        mjOBJ_BODY,
        "fr3_hand");

    link0_body_id_ = mj_name2id(
        model_copy_,
        mjOBJ_BODY,
        "fr3_link0");

    grasp_center_site_id_ = mj_name2id(
        model_copy_,
        mjOBJ_SITE,
        "tomato_attachment_center");

    if (grasp_center_site_id_ < 0)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Could not find tomato_attachment_center site.");

      return hardware_interface::CallbackReturn::ERROR;
    }

    if (hand_body_id_ < 0)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Could not find MuJoCo body fr3_hand.");

      return hardware_interface::CallbackReturn::ERROR;
    }

    if (link0_body_id_ < 0)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Could not find MuJoCo body fr3_link0.");

      return hardware_interface::CallbackReturn::ERROR;
    }

    tomatoes_.clear();

    //
    // Discover every tomato that exists in the current scene.
    //
    for (int tomato_number = 1;
         tomato_number <= 30;
         ++tomato_number)
    {

      const std::string geom_name =
          "tomato_" +
          std::to_string(tomato_number) +
          "_geom";

      const std::string body_name =
          "tomato_" +
          std::to_string(tomato_number);

      const std::string freejoint_name =
          "tomato_" +
          std::to_string(tomato_number) +
          "_freejoint";

      const int geom_id = mj_name2id(
          model_copy_,
          mjOBJ_GEOM,
          geom_name.c_str());

      const int body_id = mj_name2id(
          model_copy_,
          mjOBJ_BODY,
          body_name.c_str());

      const int freejoint_id = mj_name2id(
          model_copy_,
          mjOBJ_JOINT,
          freejoint_name.c_str());

      if (
          geom_id < 0 ||
          body_id < 0 ||
          freejoint_id < 0)
      {
        continue;
      }

      std::string stem_weld_name;
      std::string grasp_weld_name;

      if (tomato_number == 1)
      {
        stem_weld_name = "tomato_stem_weld";
        grasp_weld_name = "tomato_grasp_weld";
      }
      else
      {
        stem_weld_name =
            "tomato_" +
            std::to_string(tomato_number) +
            "_stem_weld";

        grasp_weld_name =
            "tomato_" +
            std::to_string(tomato_number) +
            "_grasp_weld";
      }

      const int stem_weld_id = mj_name2id(
          model_copy_,
          mjOBJ_EQUALITY,
          stem_weld_name.c_str());

      const int grasp_weld_id = mj_name2id(
          model_copy_,
          mjOBJ_EQUALITY,
          grasp_weld_name.c_str());

      if (stem_weld_id < 0)
      {
        RCLCPP_ERROR(
            get_logger(),
            "Tomato %d exists but stem weld '%s' "
            "was not found.",
            tomato_number,
            stem_weld_name.c_str());

        return hardware_interface::CallbackReturn::ERROR;
      }

      TomatoPhysics tomato;

      tomato.number = tomato_number;
      tomato.body_id = body_id;
      tomato.geom_id = geom_id;
      tomato.stem_weld_id = stem_weld_id;
      tomato.grasp_weld_id = grasp_weld_id;
      tomato.detached = false;
      tomato.grasp_attached = false;

      tomato.freejoint_id = freejoint_id;
      tomato.qpos_adr =
          model_copy_->jnt_qposadr[freejoint_id];

      tomatoes_.push_back(tomato);

      RCLCPP_INFO(
          get_logger(),
          "Registered tomato_%d: "
          "geom=%d, stem_weld=%d, grasp_weld=%d",
          tomato_number,
          geom_id,
          stem_weld_id,
          grasp_weld_id);
    }

    left_actuator_id_ = mj_name2id(
        model_copy_,
        mjOBJ_ACTUATOR,
        "fr3_finger_joint1");

    right_actuator_id_ = mj_name2id(
        model_copy_,
        mjOBJ_ACTUATOR,
        "fr3_finger_joint2");

    if (
        tomatoes_.empty() ||
        hand_body_id_ < 0 ||
        left_pad_geom_id_ < 0 ||
        right_pad_geom_id_ < 0 ||
        left_actuator_id_ < 0 ||
        right_actuator_id_ < 0)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Tomato harvesting physics objects "
          "were not found in the MuJoCo model.");

      return hardware_interface::CallbackReturn::ERROR;
    }

    RCLCPP_INFO(
        get_logger(),
        "Tomato harvesting physics initialized "
        "with %zu harvestable tomato(s).",
        tomatoes_.size());

    RCLCPP_INFO(
        get_logger(),
        "GelSight collision geoms: left=%d, right=%d",
        left_pad_geom_id_,
        right_pad_geom_id_);

    return hardware_interface::CallbackReturn::SUCCESS;
  }

  hardware_interface::return_type read(
      const rclcpp::Time & time,
      const rclcpp::Duration & period) override
  {
    const auto result =
        mujoco_ros2_control::MujocoSystemInterface::read(
            time,
            period);

    if (
        result !=
        hardware_interface::return_type::OK)
    {
      return result;
    }

    //
    // Once the tomato has detached, there is no need
    // to keep evaluating the detachment condition.
    //

    mjData * data_copy = nullptr;
    get_data(data_copy);

    if (data_copy == nullptr)
    {
      return result;
    }


    //
    // Look for physical contact between the tomato
    // and both GelSight collision surfaces.
    //

    const double left_command =
        data_copy->ctrl[left_actuator_id_];

    const double right_command =
        data_copy->ctrl[right_actuator_id_];

    const bool closing_command =
        left_command < 0.02 &&
        right_command < 0.02;

    const bool opening_command =
        left_command > 0.04 &&
        right_command > 0.04;

    //
    // Release any temporarily grasp-welded tomato
    // once the gripper is commanded open.
    //
    if (opening_command)
    {
      for (auto & tomato : tomatoes_)
      {
        if (
            tomato.grasp_attached &&
            tomato.grasp_weld_id >= 0)
        {
          data_copy->eq_active[
              tomato.grasp_weld_id
          ] = 0;

          tomato.grasp_attached = false;

          RCLCPP_INFO(
              get_logger(),
              "tomato_%d RELEASED from temporary "
              "gripper attachment.",
              tomato.number);
        }
      }

      //
      // Opening the gripper ends the current grasp cycle.
      // Allow a new tomato to be selected during the next closure.
      //
      active_grasp_candidate_number_ = -1;
    }


    //
    // Lock exactly one tomato when the gripper begins closing.
    // The candidate is the closest still-attached tomato to the
    // physical grasp-center site.
    //
    if (
        closing_command &&
        active_grasp_candidate_number_ < 0)
    {
      const mjtNum * grasp_center_pos =
          data_copy->site_xpos +
          3 * grasp_center_site_id_;

      double best_distance_squared = 1e9;
      int best_tomato_number = -1;

      for (const auto & tomato : tomatoes_)
      {
        if (tomato.detached)
        {
          continue;
        }

        const mjtNum * tomato_pos =
            data_copy->xpos +
            3 * tomato.body_id;

        const double dx =
            tomato_pos[0] - grasp_center_pos[0];

        const double dy =
            tomato_pos[1] - grasp_center_pos[1];

        const double dz =
            tomato_pos[2] - grasp_center_pos[2];

        const double distance_squared =
            dx * dx +
            dy * dy +
            dz * dz;

        if (distance_squared < best_distance_squared)
        {
          best_distance_squared =
              distance_squared;

          best_tomato_number =
              tomato.number;
        }
      }

      //
      // Only accept a candidate reasonably close to the
      // gripper center. 8 cm accommodates your largest
      // tomatoes while rejecting unrelated fruit.
      //

      constexpr double maximum_candidate_distance_m =
          0.12;

      if (
          best_tomato_number >= 0 &&
          best_distance_squared <=
              maximum_candidate_distance_m *
              maximum_candidate_distance_m)
      {
        active_grasp_candidate_number_ =
            best_tomato_number;

        RCLCPP_INFO(
            get_logger(),
            "Locked tomato_%d as sole grasp candidate "
            "(distance %.3f m).",
            active_grasp_candidate_number_,
            std::sqrt(best_distance_squared));
      }

    }


    //
    // Detach only when BOTH GelSight pads contact
    // the tomato while the gripper is closing.
    //
    //
    // Check every still-attached tomato independently.
    //
    for (auto & tomato : tomatoes_)
    {
      if (tomato.detached)
      {
        continue;
      }


      //
      // Only the tomato selected when closure began is
      // permitted to break its stem weld.
      //
      if (
          active_grasp_candidate_number_ < 0 ||
          tomato.number !=
              active_grasp_candidate_number_)
      {
        tomato.bilateral_contact_cycles = 0;
        continue;
      }


      bool left_contact = false;
      bool right_contact = false;
      //
      // Determine whether THIS tomato is touching
      // both GelSight pads.
      //
      for (int i = 0; i < data_copy->ncon; ++i)
      {
        const mjContact & contact =
            data_copy->contact[i];

        if (
            contact_pair_matches(
                contact,
                tomato.geom_id,
                left_pad_geom_id_))
        {
          left_contact = true;

          RCLCPP_INFO(
              get_logger(),
              "tomato_%d LEFT contact dist = %.6f m",
              tomato.number,
              contact.dist);
        }

        if (
            contact_pair_matches(
                contact,
                tomato.geom_id,
                right_pad_geom_id_))
        {
          right_contact = true;

          RCLCPP_INFO(
              get_logger(),
              "tomato_%d RIGHT contact dist = %.6f m",
              tomato.number,
              contact.dist);
        }

        if (
            left_contact &&
            right_contact)
        {
          break;
        }
      }

      if (
          left_contact &&
          right_contact &&
          closing_command)
      {
        tomato.bilateral_contact_cycles++;
      }
      else
      {
        tomato.bilateral_contact_cycles = 0;
      }

      if (
          left_contact &&
          right_contact &&
          closing_command &&
          tomato.bilateral_contact_cycles >=
              required_bilateral_contact_cycles_)
      {

        const mjtNum * tomato_world =
            data_copy->xpos + 3 * tomato.body_id;

        const mjtNum * link0_world =
            data_copy->xpos + 3 * link0_body_id_;

        const mjtNum * link0_quat =
            data_copy->xquat + 4 * link0_body_id_;

        mjtNum delta_world[3] = {
            tomato_world[0] - link0_world[0],
            tomato_world[1] - link0_world[1],
            tomato_world[2] - link0_world[2]
        };

        mjtNum link0_quat_inv[4];

        mju_negQuat(
            link0_quat_inv,
            link0_quat);

        mjtNum tomato_link0[3];

        mju_rotVecQuat(
            tomato_link0,
            delta_world,
            link0_quat_inv);

        RCLCPP_INFO(
            get_logger(),
            "GRASP DIAGNOSTIC tomato_%d in fr3_link0: "
            "x=%.6f y=%.6f z=%.6f",
            tomato.number,
            tomato_link0[0],
            tomato_link0[1],
            tomato_link0[2]);

        data_copy->eq_active[
            tomato.stem_weld_id
        ] = 0;

        if (tomato.grasp_weld_id >= 0)
        {
          data_copy->eq_active[
              tomato.grasp_weld_id
          ] = 0;

          tomato.grasp_attached = false;
        }

        //
        // Capture the exact physical grasp location.
        //
        capture_current_grasp_pose(
            tomato,
            data_copy);

        //
        // Do NOT activate the predefined equality constraint.
        //
        if (tomato.grasp_weld_id >= 0)
        {
          data_copy->eq_active[
              tomato.grasp_weld_id
          ] = 0;
        }

        tomato.grasp_attached = true;

        tomato.detached = true;
        tomato.bilateral_contact_cycles = 0;

        RCLCPP_INFO(
            get_logger(),
            "tomato_%d DETACHED after sustained "
            "bilateral GelSight grasp.",
            tomato.number);

        break;
      }
    }




    //
    // Carry grasped tomatoes at exactly the relative pose
    // captured at the moment of stable bilateral contact.
    //
    for (auto & tomato : tomatoes_)
    {
      if (
          tomato.detached &&
          tomato.grasp_attached)
      {
        apply_temporary_grasp_attachment(
            tomato,
            data_copy);
      }
    }

    set_data(data_copy);

    mj_deleteData(data_copy);

    return result;
  }

private:

  struct TomatoPhysics
  {
    int number{-1};

    int body_id{-1};
    int geom_id{-1};

    int freejoint_id{-1};
    int qpos_adr{-1};

    int stem_weld_id{-1};
    int grasp_weld_id{-1};

    bool detached{false};
    bool grasp_attached{false};

    int bilateral_contact_cycles{0};

    // Tomato pose relative to fr3_hand at the instant
    // the stable bilateral grasp is confirmed.
    mjtNum grasp_rel_pos[3]{0, 0, 0};

    // MuJoCo quaternion ordering: w, x, y, z.
    mjtNum grasp_rel_quat[4]{1, 0, 0, 0};
  };

  std::vector<TomatoPhysics> tomatoes_;

  static constexpr int required_bilateral_contact_cycles_ = 20;

  static bool contact_pair_matches(
      const mjContact & contact,
      int geom_a,
      int geom_b)
  {
    return (
        (
            contact.geom1 == geom_a &&
            contact.geom2 == geom_b
        ) ||
        (
            contact.geom1 == geom_b &&
            contact.geom2 == geom_a
        )
    );
  }

  void capture_current_grasp_pose(
      TomatoPhysics & tomato,
      const mjData * data)
  {
    const mjtNum * hand_pos =
        data->xpos + 3 * hand_body_id_;

    const mjtNum * hand_quat =
        data->xquat + 4 * hand_body_id_;

    const mjtNum * tomato_pos =
        data->xpos + 3 * tomato.body_id;

    const mjtNum * tomato_quat =
        data->xquat + 4 * tomato.body_id;

    //
    // World-space vector from hand to tomato.
    //
    mjtNum delta_world[3] = {
        tomato_pos[0] - hand_pos[0],
        tomato_pos[1] - hand_pos[1],
        tomato_pos[2] - hand_pos[2]
    };

    //
    // Express that vector in the hand frame.
    //
    mjtNum hand_quat_inv[4];

    mju_negQuat(
        hand_quat_inv,
        hand_quat);

    mju_rotVecQuat(
        tomato.grasp_rel_pos,
        delta_world,
        hand_quat_inv);

    //
    // Relative orientation:
    //
    // q_rel = inverse(q_hand) * q_tomato
    //
    mju_mulQuat(
        tomato.grasp_rel_quat,
        hand_quat_inv,
        tomato_quat);

    mju_normalize4(
        tomato.grasp_rel_quat);

    RCLCPP_INFO(
        get_logger(),
        "Captured tomato_%d grasp-relative pose: "
        "xyz=[%.4f %.4f %.4f].",
        tomato.number,
        tomato.grasp_rel_pos[0],
        tomato.grasp_rel_pos[1],
        tomato.grasp_rel_pos[2]);
  }

  void apply_temporary_grasp_attachment(
      TomatoPhysics & tomato,
      mjData * data)
  {
    if (
        tomato.qpos_adr < 0 ||
        tomato.freejoint_id < 0 ||
        hand_body_id_ < 0)
    {
      return;
    }

    const mjtNum * hand_pos =
        data->xpos + 3 * hand_body_id_;

    const mjtNum * hand_quat =
        data->xquat + 4 * hand_body_id_;

    //
    // Desired tomato position from the grasp pose
    // captured at initial bilateral contact.
    //
    mjtNum offset_world[3];

    mju_rotVecQuat(
        offset_world,
        tomato.grasp_rel_pos,
        hand_quat);

    mjtNum target_pos[3] = {
        hand_pos[0] + offset_world[0],
        hand_pos[1] + offset_world[1],
        hand_pos[2] + offset_world[2]
    };

    //
    // Desired tomato orientation.
    //
    mjtNum target_quat[4];

    mju_mulQuat(
        target_quat,
        hand_quat,
        tomato.grasp_rel_quat);

    mju_normalize4(
        target_quat);

    const int adr = tomato.qpos_adr;

    //
    // COMPLIANT temporary attachment.
    //
    // Do not teleport the tomato directly onto the target.
    // Move only part of the error each control cycle so
    // MuJoCo contact forces can resolve GelSight penetration.
    //
    constexpr mjtNum attachment_gain = 0.12;

    for (int i = 0; i < 3; ++i)
    {
      data->qpos[adr + i] +=
          attachment_gain *
          (
              target_pos[i] -
              data->qpos[adr + i]
          );
    }

    //
    // Quaternion interpolation.
    // Ensure target quaternion is on the same hemisphere.
    //
    mjtNum current_quat[4] = {
        data->qpos[adr + 3],
        data->qpos[adr + 4],
        data->qpos[adr + 5],
        data->qpos[adr + 6]
    };

    mjtNum quat_dot =
        current_quat[0] * target_quat[0] +
        current_quat[1] * target_quat[1] +
        current_quat[2] * target_quat[2] +
        current_quat[3] * target_quat[3];

    if (quat_dot < 0.0)
    {
      for (int i = 0; i < 4; ++i)
      {
        target_quat[i] = -target_quat[i];
      }
    }

    mjtNum blended_quat[4];

    for (int i = 0; i < 4; ++i)
    {
      blended_quat[i] =
          (1.0 - attachment_gain) *
              current_quat[i] +
          attachment_gain *
              target_quat[i];
    }

    mju_normalize4(
        blended_quat);

    data->qpos[adr + 3] = blended_quat[0];
    data->qpos[adr + 4] = blended_quat[1];
    data->qpos[adr + 5] = blended_quat[2];
    data->qpos[adr + 6] = blended_quat[3];

    //
    // Damp, but do not completely erase, tomato velocity.
    // Contact solver can therefore still separate the
    // tomato from the GelSight collision surfaces.
    //
    const int dof_adr =
        model_copy_->jnt_dofadr[
            tomato.freejoint_id
        ];

    constexpr mjtNum velocity_damping = 0.80;

    for (int i = 0; i < 6; ++i)
    {
      data->qvel[dof_adr + i] *=
          velocity_damping;
    }
  }

  mjModel * model_copy_{nullptr};

  int hand_body_id_{-1};

  int link0_body_id_{-1};

  int grasp_center_site_id_{-1};

  int active_grasp_candidate_number_{-1};


  int left_pad_geom_id_{-1};
  int right_pad_geom_id_{-1};

  int left_actuator_id_{-1};
  int right_actuator_id_{-1};

};

}  // namespace mujoco_fr3_bringup


PLUGINLIB_EXPORT_CLASS(
    mujoco_fr3_bringup::FR3TomatoMujocoSystem,
    hardware_interface::SystemInterface)
