import numpy as np
import random

class TimeManagementEnv:
    def __init__(self, tasks, total_time):
        self.tasks = tasks
        self.total_time = total_time
        self.reset()
    
    def reset(self):
        # Initialize state: remaining time and task statuses (0: incomplete, 1: complete)
        self.state = [self.total_time, [0] * len(self.tasks)]
        return self.state

    def step(self, action):
        task = self.tasks[action]
        task_time = task['time']
        task_priority = task['priority']
        
        # Check if the task is valid (not completed and within available time)
        if self.state[0] >= task_time and self.state[1][action] == 0:
            # Update remaining time and mark task as completed
            self.state[0] -= task_time
            self.state[1][action] = 1
            reward = task_priority  # Reward is based on task priority
            done = all(self.state[1])  # Check if all tasks are completed
        else:
            # Invalid action: Task already completed or insufficient time
            reward = -1  # Penalty for invalid action
            done = False
        
        return self.state, reward, done


def rl_time_management():
    # Input number of tasks
    n = int(input("Enter the number of tasks: "))
    
    tasks = []
    total_time = 0  # Sum up total time required for all tasks
    for i in range(n):
        print(f"\nTask {i+1}:")
        name = input("Enter task name: ")
        priority = int(input("Enter task priority (higher is better): "))
        time = float(input("Enter time required to complete (in hours): "))
        tasks.append({"name": name, "priority": priority, "time": time})
        total_time += time

    # Initialize environment
    env = TimeManagementEnv(tasks, total_time)

    # Q-Learning parameters
    q_table = np.zeros((int(total_time) + 1, len(tasks)))  # State: Remaining time, Actions: Tasks
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 1000

    # Train the RL agent
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            remaining_time = state[0]
            task_status = tuple(state[1])

            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(len(tasks)))
            else:
                # Select the best action from Q-table
                action = np.argmax(q_table[int(remaining_time), :])

            # Take action and observe the result
            next_state, reward, done = env.step(action)

            # Update Q-table using Q-Learning formula
            next_remaining_time = next_state[0]
            best_next_action = np.argmax(q_table[int(next_remaining_time), :])
            q_table[int(remaining_time), action] += alpha * (
                reward + gamma * q_table[int(next_remaining_time), best_next_action] - q_table[int(remaining_time), action]
            )

            state = next_state

    # Evaluate the trained policy to determine the task order
    state = env.reset()
    done = False
    recommended_order = []
    visited_tasks = set()

    while not done:
        remaining_time = state[0]

        # Filter available actions (tasks that are incomplete and within time)
        available_actions = [i for i in range(len(tasks)) if i not in visited_tasks and state[1][i] == 0]

        if not available_actions:
            break  # No more valid actions

        # Choose the best action based on Q-table
        action_values = [q_table[int(remaining_time), a] for a in available_actions]
        action = available_actions[np.argmax(action_values)]

        # Take the selected action
        next_state, _, done = env.step(action)
        visited_tasks.add(action)
        recommended_order.append(tasks[action]['name'])

        state = next_state

    # Output the recommended task order
    print("\nRecommended Task Order:")
    for i, task_name in enumerate(recommended_order, start=1):
        print(f"{i}. {task_name}")


if __name__ == "__main__":
    rl_time_management()
3