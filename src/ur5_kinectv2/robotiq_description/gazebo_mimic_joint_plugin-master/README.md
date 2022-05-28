# gazebo_mimic_plugin

Gazebo의 경우 URDF에서 생성한 mimic_control에 대해 지원을 하지 않음. 따라서 이를 이용하기 위한 플러그인이 필요함.

Reference: https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins

사용예)

    <xacro:macro name="insert_mimic_joint" params="reference name multiplier offset">
        <gazebo>
            <plugin name="mimic_${name}" filename="libgazebo_mimic_joint_plugin.so">
                <joint>${reference}</joint>
                <mimicJoint>${name}</mimicJoint>
                <multiplier>${multiplier}</multiplier>
                <offset>${offset}</offset>
            </plugin>
        </gazebo>
    </xacro:macro>


    <xacro:insert_mimic_joint reference="ref_joint_name" name="mimic_joint_name" multiplier="1.0" offset="1.0"/>