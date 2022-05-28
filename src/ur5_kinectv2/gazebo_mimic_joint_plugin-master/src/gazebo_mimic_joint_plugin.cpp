#include <gazebo_mimic_joint_plugin/gazebo_mimic_joint_plugin.h>

namespace math = ignition::math;
namespace gazebo {

    GazeboMimicJointPlugin::GazeboMimicJointPlugin()
    {
        joint_.reset();
        mimic_joint_.reset();
    }

    GazeboMimicJointPlugin::~GazeboMimicJointPlugin()
    {
        this->updateConnection.reset();
    }

    void GazeboMimicJointPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
        ros::NodeHandle model_nh;
        model_ = _parent;
        world_ = model_->GetWorld();

        // Error message if the model couldn't be found
        if (!model_) {
            ROS_ERROR("Parent model is NULL! GazeboNaoqiControlPlugin could not be loaded.");
            return;
        }

        // Check that ROS has been initialized
        if (!ros::isInitialized()) {
            ROS_ERROR("A ROS node for Gazebo has not been initialized, unable to load plugin.");
            return;
        }

        // Check for robot namespace
        robot_namespace_ = "/";
        if (_sdf->HasElement("robotNamespace")) {
            robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
        }

        // Check for joint element
        if (!_sdf->HasElement("joint")) {
            ROS_ERROR("No joint element present. GazeboMimicJointPlugin could not be loaded.");
            return;
        }

        joint_name_ = _sdf->GetElement("joint")->Get<std::string>();

        // Check for mimicJoint element
        if (!_sdf->HasElement("mimicJoint")) {
            ROS_ERROR("No mimicJoint element present. GazeboMimicJointPlugin could not be loaded.");
            return;
        }

        mimic_joint_name_ = _sdf->GetElement("mimicJoint")->Get<std::string>();

        has_pid_ = false;
        // Check if PID controller wanted
        if (_sdf->HasElement("hasPID")) {
            has_pid_ = true;

            const ros::NodeHandle nh(model_nh, std::string(robot_namespace_ + "/gazebo_ros_control/pid_gains/") + mimic_joint_name_);
            pid_.init(nh);
        }

        // Check for multiplier element
        multiplier_ = 1.0;
        if (_sdf->HasElement("multiplier"))
            multiplier_ = _sdf->GetElement("multiplier")->Get<double>();

        // Check for offset element
        offset_ = 0.0;
        if (_sdf->HasElement("offset"))
            offset_ = _sdf->GetElement("offset")->Get<double>();

        // Check for sensitiveness element
        sensitiveness_ = 0.0;
        if (_sdf->HasElement("sensitiveness"))
            sensitiveness_ = _sdf->GetElement("sensitiveness")->Get<double>();

        // Check for max effort
        max_effort_ = 1.0;
        if (_sdf->HasElement("maxEffort")) {
            max_effort_ = _sdf->GetElement("maxEffort")->Get<double>();
        }

        // Get pointers to joints
        joint_ = model_->GetJoint(joint_name_);
        if (!joint_) {
            ROS_ERROR_STREAM("No joint named \"" << joint_name_ << "\". GazeboMimicJointPlugin could not be loaded.");
            return;
        }
        mimic_joint_ = model_->GetJoint(mimic_joint_name_);
        if (!mimic_joint_) {
            ROS_ERROR_STREAM("No (mimic) joint named \"" << mimic_joint_name_ << "\". GazeboMimicJointPlugin could not be loaded.");
            return;
        }

        // Set max effort
        if (!has_pid_) {
            mimic_joint_->SetEffortLimit(0, max_effort_);
        }

        // Listen to the update event. This event is broadcast every
        // simulation iteration.
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
            boost::bind(&GazeboMimicJointPlugin::UpdateChild, this));

        // Output some confirmation
        ROS_INFO_STREAM("GazeboMimicJointPlugin loaded! Joint: \"" << joint_name_ << "\", Mimic joint: \"" << mimic_joint_name_ << "\""
                                                             << ", Multiplier: " << multiplier_ << ", Offset: " << offset_
                                                             << ", MaxEffort: " << max_effort_ << ", Sensitiveness: " << sensitiveness_);
    }

    void GazeboMimicJointPlugin::UpdateChild()
    {
        static ros::Duration period(world_->Physics()->GetMaxStepSize());

        // Set mimic joint's angle based on joint's angle
        double angle = joint_->Position(0) * multiplier_ + offset_;
        double a = mimic_joint_->Position(0);

        if (abs(angle - a) >= sensitiveness_) {
            if (has_pid_) {
                if (a != a)
                    a = angle;
                double error = angle - a;
                double effort = math::clamp(pid_.computeCommand(error, period), -max_effort_, max_effort_);
                mimic_joint_->SetForce(0, effort);
            }
            else {
                mimic_joint_->SetPosition(0, angle, true);
            }
        }
    }

    GZ_REGISTER_MODEL_PLUGIN(GazeboMimicJointPlugin);
}