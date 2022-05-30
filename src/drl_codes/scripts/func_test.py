    def sim_grasp(self):
        object_names, objects = self.get_model_states()
        not_separated_models = set()
        separate = 0
        for i in range(len(objects)):
            if_separate = True
            for j in range(i + 1, len(objects)):
                distance = (objects[i][0] - objects[j][0]) ** 2 + (objects[i][1] - objects[j][1]) ** 2
                if distance > 0.04:
                    separate += 1
                else:
                    if_separate = False
            if not if_separate:
                not_separated_models.add(i)
                not_separated_models.add(j)
        
        separated_models = set([i for i in range(len(objects))]) - not_separated_models
        for i in separated_models:
            rospy.wait_for_service('/gazebo/delete_model', timeout=5)
            self.delete_model(object_names[i])
        return separate, separate - (len(objects) - 1) * len(separated_models)