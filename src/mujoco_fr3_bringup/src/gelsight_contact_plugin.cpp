#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp>




namespace mujoco_fr3_bringup
{

class GelSightContactPlugin
    : public mujoco_ros2_control_plugins::
          MuJoCoROS2ControlPluginBase
{
public:
  GelSightContactPlugin() = default;

  ~GelSightContactPlugin() override = default;

  bool init(
      rclcpp::Node::SharedPtr node,
      const mjModel * model,
      mjData * data) override
  {
    (void)data;

    node_ = node;

    // --------------------------------------------------------
    // Find ALL generated tomato geoms.
    //
    // Accepted names:
    //   tomato_1_geom
    //   tomato_2_geom
    //   ...
    //
    // This removes the old single-tomato assumption.
    // --------------------------------------------------------

    tomato_geom_ids_.clear();

    for (int geom_id = 0; geom_id < model->ngeom; ++geom_id)
    {
      const char * geom_name =
          mj_id2name(
              model,
              mjOBJ_GEOM,
              geom_id);

      if (geom_name == nullptr)
      {
        continue;
      }

      const std::string name(
          geom_name);

      if (
          name.rfind("tomato_", 0) == 0 &&
          name.size() >= 5 &&
          name.compare(
              name.size() - 5,
              5,
              "_geom") == 0)
      {
        tomato_geom_ids_.push_back(
            geom_id);
      }
    }

    left_geom_id_ = mj_name2id(
        model,
        mjOBJ_GEOM,
        "gelsight_left_collision");

    right_geom_id_ = mj_name2id(
        model,
        mjOBJ_GEOM,
        "gelsight_right_collision");

    if (
        tomato_geom_ids_.empty() ||
        left_geom_id_ < 0 ||
        right_geom_id_ < 0)
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "GelSightContactPlugin could not find "
          "required MuJoCo geoms.");

      return false;
    }

    publisher_ =
        node_->create_publisher<
            std_msgs::msg::Float64MultiArray>(
            "/gelsight/contact_data",
            10);

    RCLCPP_INFO(
        node_->get_logger(),
        "GelSightContactPlugin initialized: "
        "%zu tomato geoms, left=%d right=%d",
        tomato_geom_ids_.size(),
        left_geom_id_,
        right_geom_id_);

    return true;
  }


  bool is_tomato_geom(
      int geom_id) const
  {
    return (
        std::find(
            tomato_geom_ids_.begin(),
            tomato_geom_ids_.end(),
            geom_id)
        != tomato_geom_ids_.end());
  }

  void update(
      const mjModel * model,
      mjData * data) override
  {
    (void)model;

    bool left_contact = false;
    bool right_contact = false;

    double left_indentation = 0.0;
    double right_indentation = 0.0;

    //
    // Accumulators used to construct a stable effective
    // contact from all MuJoCo contact points.
    //
    double left_weight_sum = 0.0;
    double right_weight_sum = 0.0;

    double left_indentation_sum = 0.0;
    double right_indentation_sum = 0.0;

    std::array<double, 3> left_position_sum{
        0.0, 0.0, 0.0};

    std::array<double, 3> right_position_sum{
        0.0, 0.0, 0.0};

    std::array<double, 3> left_local{
        0.0, 0.0, 0.0};

    std::array<double, 3> right_local{
        0.0, 0.0, 0.0};

    //
    // Inspect every MuJoCo contact.
    //
    for (int i = 0; i < data->ncon; ++i)
    {
      const mjContact & contact =
          data->contact[i];

      //
      // LEFT GEL
      //
      if (
          (
              contact.geom1 == left_geom_id_ &&
              is_tomato_geom(contact.geom2)
          )
          ||
          (
              contact.geom2 == left_geom_id_ &&
              is_tomato_geom(contact.geom1)
          )
      )
      {
        left_contact = true;

        const double indentation =
            std::max(
                0.0,
                -static_cast<double>(
                    contact.dist));

        std::array<double, 3> local{
            0.0, 0.0, 0.0};

        world_to_geom_local(
            data,
            left_geom_id_,
            contact.pos,
            local);

        //
        // Give deeper contacts more influence while
        // still allowing shallow contacts to contribute.
        //
        const double weight =
            std::max(
                indentation,
                1e-6);

        left_weight_sum += weight;

        left_position_sum[0] +=
            weight * local[0];

        left_position_sum[1] +=
            weight * local[1];

        left_position_sum[2] +=
            weight * local[2];

        left_indentation_sum +=
            weight * indentation;
      }

      //
      // RIGHT GEL
      //
      if (
          (
              contact.geom1 == right_geom_id_ &&
              is_tomato_geom(contact.geom2)
          )
          ||
          (
              contact.geom2 == right_geom_id_ &&
              is_tomato_geom(contact.geom1)
          )
      )
      {
        right_contact = true;

        const double indentation =
            std::max(
                0.0,
                -static_cast<double>(
                    contact.dist));

        std::array<double, 3> local{
            0.0, 0.0, 0.0};

        world_to_geom_local(
            data,
            right_geom_id_,
            contact.pos,
            local);

        const double weight =
            std::max(
                indentation,
                1e-6);

        right_weight_sum += weight;

        right_position_sum[0] +=
            weight * local[0];

        right_position_sum[1] +=
            weight * local[1];

        right_position_sum[2] +=
            weight * local[2];

        right_indentation_sum +=
            weight * indentation;
      }
    }

    //
    // Convert all LEFT contacts into one stable,
    // penetration-weighted effective contact.
    //
    if (
        left_contact &&
        left_weight_sum > 0.0)
    {
      left_local[0] =
          left_position_sum[0]
          / left_weight_sum;

      left_local[1] =
          left_position_sum[1]
          / left_weight_sum;

      left_local[2] =
          left_position_sum[2]
          / left_weight_sum;

      left_indentation =
          left_indentation_sum
          / left_weight_sum;
    }

    //
    // Convert all RIGHT contacts into one stable,
    // penetration-weighted effective contact.
    //
    if (
        right_contact &&
        right_weight_sum > 0.0)
    {
      right_local[0] =
          right_position_sum[0]
          / right_weight_sum;

      right_local[1] =
          right_position_sum[1]
          / right_weight_sum;

      right_local[2] =
          right_position_sum[2]
          / right_weight_sum;

      right_indentation =
          right_indentation_sum
          / right_weight_sum;
    }

    std_msgs::msg::Float64MultiArray msg;

    //
    // Message layout:
    //
    // [0] left contact: 0/1
    // [1] left local x [m]
    // [2] left local y [m]
    // [3] left local z [m]
    // [4] left effective indentation [m]
    //
    // [5] right contact: 0/1
    // [6] right local x [m]
    // [7] right local y [m]
    // [8] right local z [m]
    // [9] right effective indentation [m]
    //
    msg.data = {
        left_contact ? 1.0 : 0.0,
        left_local[0],
        left_local[1],
        left_local[2],
        left_indentation,

        right_contact ? 1.0 : 0.0,
        right_local[0],
        right_local[1],
        right_local[2],
        right_indentation,
    };

    publisher_->publish(msg);
  }

  void cleanup() override
  {
    publisher_.reset();
    node_.reset();
  }

private:
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

  static void world_to_geom_local(
      const mjData * data,
      int geom_id,
      const mjtNum world[3],
      std::array<double, 3> & local)
  {
    const mjtNum * geom_pos =
        data->geom_xpos + 3 * geom_id;

    const mjtNum * geom_rot =
        data->geom_xmat + 9 * geom_id;

    const double dx =
        static_cast<double>(
            world[0] - geom_pos[0]);

    const double dy =
        static_cast<double>(
            world[1] - geom_pos[1]);

    const double dz =
        static_cast<double>(
            world[2] - geom_pos[2]);

    //
    // MuJoCo xmat is the local-to-world rotation
    // matrix, so R^T converts the world displacement
    // into the collision geom's local frame.
    //
    local[0] =
        geom_rot[0] * dx +
        geom_rot[3] * dy +
        geom_rot[6] * dz;

    local[1] =
        geom_rot[1] * dx +
        geom_rot[4] * dy +
        geom_rot[7] * dz;

    local[2] =
        geom_rot[2] * dx +
        geom_rot[5] * dy +
        geom_rot[8] * dz;
  }

  rclcpp::Node::SharedPtr node_;

  rclcpp::Publisher<
      std_msgs::msg::Float64MultiArray>::
      SharedPtr publisher_;

  std::vector<int> tomato_geom_ids_;
  int left_geom_id_{-1};
  int right_geom_id_{-1};
};

}  // namespace mujoco_fr3_bringup


PLUGINLIB_EXPORT_CLASS(
    mujoco_fr3_bringup::GelSightContactPlugin,
    mujoco_ros2_control_plugins::
        MuJoCoROS2ControlPluginBase)
