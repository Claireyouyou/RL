from maze import Maze
from RL_Brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()  # 一行一行自己打会出现Unresolved reference 'env'。在调用里定义了

        while True:
            # fresh env
            env.render()

            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_))

            observation_ = observation

            if done:
                break
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()

