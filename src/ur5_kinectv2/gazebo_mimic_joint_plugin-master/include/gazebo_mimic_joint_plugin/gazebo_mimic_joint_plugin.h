#ifndef GAZEBO_MIMIC_JOINT_PLUGIN_H_
#define GAZEBO_MIMIC_JOINT_PLUGIN_H_

#include <ros/ros.h>
#include <control_toolbox/pid.h>
#include <boost/bind.hpp>

#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

namespace gazebo {
    class GazeboMimicJointPlugin : public ModelPlugin {
    public:
        GazeboMimicJointPlugin();
        ~GazeboMimicJointPlugin();

        void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf);
        void UpdateChild();

    private:
        std::string joint_name_, mimic_joint_name_, robot_namespace_;
        double multiplier_, offset_, sensitiveness_, max_effort_;
        bool has_pid_;

        control_toolbox::Pid pid_;
        physics::JointPtr joint_, mimic_joint_;
        physics::ModelPtr model_;
        physics::WorldPtr world_;

        event::ConnectionPtr updateConnection;
    };
}

#endif