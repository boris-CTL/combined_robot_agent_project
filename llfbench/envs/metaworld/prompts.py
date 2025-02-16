
mw_instruction = (
    "Your job is to control a Sawyer robot arm to solve a {task} task. You will get observations of the robot state and the world state in the form of json strings. Your objective is to provide control inputs to the robot to achieve the task's goal state over multiple time steps. Your actions are 4-dim vectors, where the first 3 dimensions control the movement of the robot's end effector in the x, y, and z directions, and the last dimension controls the gripper state (0 means opening it, and 1 means closing it). You action at each step sets the robot's target pose for that step. The robot will move towards that pose using a P controller.",
)


# this b_intruction is added by myself. boris.
b_instruction = (
    "Output a good action (4-dim vector) in the form of [x_input, y_input, z_input, gripper_state].",
    "Render a proper action (4-dim vector) in the style of [x_input, y_input, z_input, gripper_state].",
    "Supply a good action (4-dim vector) in the form of [x_input, y_input, z_input, gripper_state].",
    "Indicate a good action (4-dim vector) using [x_input, y_input, z_input, gripper_state] as the format.",
    "Come up with a good action (4-dim vector) in the form of [x_input, y_input, z_input, gripper_state].",
    "Present a correct action (4-dim vector) in the form of [x_input, y_input, z_input, gripper_state].",
)


r_feedback = (
    "Your reward for the latest step is {reward}.",
    "You got a reward of {reward}.",
    "The latest step brought you {reward} reward units.",
    "You've received a reward of {reward}.",
    "You've earned a reward of {reward}.",
    "You just got {reward} points.",
    "{reward} points for you.",
    "You've got yourself {reward} units of reward.",
    "The reward your latest step earned you is {reward}.",
    "The previous step's reward was {reward}.",
    "+{reward} reward",
    "Your reward is {reward}.",
    "The reward you just earned is {reward}.",
    "You have received {reward} points of reward.",
    "Your reward={reward}.",
    "The reward is {reward}.",
    "Alright, you just earned {reward} reward units.",
    "Your instantaneous reward is {reward}.",
    "Your rew. is {reward}.",
    "+{reward} points",
    "Your reward gain is {reward}."
)

# fp_feedback = (
#     "You should go to {expert_action}.",
#     "I recommend that you move to {expert_action}.",
#     "Move to {expert_action}.",
#     "I suggest you move to pose {expert_action}.",
#     "Assuming pose {expert_action} will help you get to the goal faster.",
#     "Try moving to pose {expert_action}.",
#     "One thing to try is to take action {expert_action}.",
#     "Action {expert_action} is promising.",
#     "Aim to reach pose {expert_action} at the next step.",
#     "My advice is to take action {expert_action}.",
#     "Go for action {expert_action}.",
#     "I would try action {expert_action} if I were you.",
#     "Consider going to {expert_action}.",
#     "Attempt to reach pose {expert_action} next.",
#     "My suggestion is that you go towards pose {expert_action}.",
#     "Moving to pose {expert_action} next looks promising.",
#     "I advise you to take action {expert_action}.",
#     "Next, move to pose {expert_action}.",
#     "Moving to {expert_action} now is a good idea.",
#     "If you want a tip, {expert_action} is a good pose to aim for next.",
#     "I urge you to move to pose {expert_action}.",
# )


fp_feedback = (
    " ",
    "  "
)


# open_gripper_feedback = (
#     "You should open the gripper.",
#     "The gripper needs to be opened.",
#     "It's necessary to open the gripper.",
#     "Open the gripper.",
#     "You need to open the gripper.",
#     "You ought to open the gripper.",
#     "You must open the gripper.",
#     "You have to open the gripper.",
#     "Consider opening the gripper.",
#     "You're supposed to open the gripper.",
#     "You'll want to open the gripper.",
#     "Opening the gripper now is a good idea.",
#     "You'll need to open the gripper.",
#     "The gripper should be opened.",
#     "Try opening the gripper.",
#     "I advise you to open the gripper.",
#     "Remember to open the gripper.",
#     "I suggest you open the gripper.",
#     "The gripper must be open.",
#     "Don't forget to open the gripper.",
#     "If you don't open the gripper you'll have an issue.",
# )


open_gripper_feedback = (
    " ",
    "  "
)




# close_gripper_feedback = (
#     "You should close the gripper.",
#     "You need to close the gripper.",
#     "It's necessary to close the gripper.",
#     "Close the gripper.",
#     "Consider closing the gripper.",
#     "You must close the gripper.",
#     "You will want to close the gripper.",
#     "It's essential that you close the gripper.",
#     "You're supposed to close the gripper.",
#     "You have to close the gripper.",
#     "You'll need to close the gripper.",
#     "Closing the gripper at this point is essential.",
#     "Your gripper needs to be shut.",
#     "The gripper needs to be closed.",
#     "You're supposed to close down the gripper.",
#     "Try closing the gripper.",
#     "Closing the gripper now would be good.",
#     "The gripper ought to be closed.",
#     "Keep the gripper closed.",
#     "Closing the gripper at this point is a must.",
#     "Your gripper should be closed.",
# )




close_gripper_feedback = (
    " ",
    "  "
)



# hp_feedback = (
#     "You're getting closer. Keep going!",
#     "You are making headway. Don't stop!",
#     "You're on the right path. Continue!",
#     "You are making progress towards achieving the goal. Keep it up!",
#     "You're almost there. Keep moving!",
#     "You're heading the right way. Go on!",
#     "You're heading in the right direction.",
#     "Getting there! Keep advancing!",
#     "You are on the correct path.",
#     "Your latest move has got you closer to finishing your task.",
#     "You are making good progress.",
#     "Good going! You are getting closer to completing the task.",
#     "Your progress is great! Continue down the same path.",
#     "You are on the right track. Keep it up!",
#     "You are moving the right way.",
#     "You are moving the arm in the correct direction.",
#     "Keep progressing! Your latest move has been great.",
#     "You are getting there! You are moving in the right general direction.",
#     "You are advacing towards completing the task. Keep making progress!",
#     "You're closing in on your goal. Keep at it!",
#     "Press on! You've moving in the right direction.",
# )

hp_feedback = (
    " ",
    "  "
)


# hn_feedback = (
#     "You moved in the wrong direction. You're now farther away from the goal than before.",
#     "You've gone in the incorrect direction. Your distance to the target has increased.",
#     "You've moved away from the goal.",
#     "You've moved the wrong way. You are presently farther from the goal than before.",
#     "You are moving the wrong way. The goal is now farther than previously.",
#     "Wrong direction. You're currently farther from achieving the goal than earlier.",
#     "You've strayed in the wrong direction. The target is now farther away.",
#     "Your move direction is off. Getting to your objective will now take longer.",
#     "You've drifted in the incorrect direction. You're now farther from the completing the task than earlier.",
#     "You've deviated from the right path and have moved away from the goal.",
#     "You're now farther from the target than previously. You've moved the wrong way.",
#     "Your latest action has taken you further away from completing your task.",
#     "The action you have just taken has moved the arm in the wrong direction.",
#     "The arm has made a highly suboptimal move. You are currently farther away from the goal than previously.",
#     "You've moved off the right path. You are now farther from the target than earlier.",
#     "The direction of the arm's latest move is wrong. Achieving the goal will now take longer than before.",
#     "The arm's move direction is off. Compared to the previous state, you are now farther from finishing the task.",
#     "The latest action has taken you the wrong way. You're currently further from the goal than earlier.",
#     "You've vectored the arm in a direction that will make it more difficult to accomplish your task.",
#     "The latest arm movement was in a wrong direction. Finishing the task is now more distant than previously.",
#     "As a result of your latest move, you are now further from your objective.",
# )


hn_feedback = (
    " "
)