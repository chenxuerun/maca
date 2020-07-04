'''
        friends_big_map = to_big_map(alive_friend_coordinates)
        enemys_big_map = to_big_map(enemy_coordinates)

                attack = 0
                friend_big_map, big_block = to_big_map(alive_friend_coordinate)
                big_block_net_input_array = np.stack(
                    [friends_big_map, enemys_big_map, friend_big_map], axis=0)
                big_block_net_input_tensor = torch.Tensor(big_block_net_input_array).unsqueeze_(0)

                big_block_net_out_tensor = self.big_block_net(big_block_net_input_tensor)[0]
                big_block_net_out_array = big_block_net_out_tensor.detach().numpy()
                big_block_max_index = np.argmax(big_block_net_out_array)

                if equal(block=big_block, number=big_block_max_index):
                    friends_small_map = to_small_map(alive_friend_coordinates)
                    enemys_small_map = to_small_map(enemy_coordinates)
                    friend_small_map, small_block = to_small_map(alive_friend_coordinate)
                    small_block_net_input_array = np.stack(
                        [friends_small_map, enemys_small_map, friend_small_map], axis=0)
                    small_block_net_input_tensor = torch.Tensor(small_block_net_input_array).unsqueeze_(0)

                    small_block_net_out_tensor = self.small_block_net(small_block_net_input_tensor)[0]
                    small_block_net_out_array = small_block_net_out_tensor.detach().numpy()
                    small_block_max_index = np.argmax(small_block_net_out_array)

                    if equal(block=small_block, number=small_block_max_index):
                        course = random.randint(0, 359)
                    else:
                        goal = block_center_coordinate(big_block=big_block, small_block=small_block)
                        course = course_start_to_goal(start=alive_friend_coordinate, goal=goal)
                else:
                    goal = block_center_coordinate(big_block=big_block)
                    course = course_start_to_goal(start=alive_friend_coordinate, goal=goal)
'''

'''
            for j, enemy_index in enumerate(sorted_enemy_index): # 看看有没有目标
                enemy_id = sorted_enemy_ids[j]
                # 已经被2颗炮弹打击，不管它了
                if enemy_striked_num[enemy_id - 1] >= 2: continue

                # 一旦不足2颗炮弹，就再不看后面的了
                distance = sorted_enemy_distances[j]
                if distance < SHORT_MISSLE_RANGE:
                    if alive_friend_inf[S_MISSILE_LEFT]:
                        attack = enemy_id + 10
                    elif alive_friend_inf[L_MISSILE_LEFT]:
                        attack = enemy_id
                    goal = new_enemy_coordinates[enemy_index]
                elif distance < LONG_MISSLE_RANGE:
                    if alive_friend_inf[L_MISSILE_LEFT]:
                        attack = enemy_id
                    goal = new_enemy_coordinates[enemy_index]
                elif distance < ALERT_RANGE:
                    goal = new_enemy_coordinates[enemy_index]
                break
'''