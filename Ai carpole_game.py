import gym
import random
import numpy as np
from statistics import mean, median
from tqdm import tqdm
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split




env = gym.make('CartPole-v0')

initial_games = 1000
goal_steps = 500
score_requirement = 50


def random_games():
    for i in range(5):
        observation = env.reset()
        for _ in range(200):
            #  This will display the environment
            env.render()
            # This will just create random output 0 or 1
            print(observation, '\n')
            action = env.action_space.sample()
            observation, reward, info, done = env.step(action)

            if done:
                break
    env.close()


def init_population():
    training_data = []  # observation and moves
    scores = []
    accepted_score = []  # just the score greater then 50

    for _ in tqdm(range(initial_games)):
        observation = env.reset()
        ob = observation
        score = 0
        game_memory = []  # move from the environment
        prev_observation = []  # previous observation
        for _ in range(goal_steps):

            # game

            action = random.randrange(0, 2) # action between 0, 1
            observation, reward, info, done = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])  # append data in game memory

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            #  append all score greater then
            accepted_score.append(score)
            for data in game_memory:
                # making data one hot
                if data[1] == 1:
                    output = [1]
                else:
                    output = [0]
                # adding prev_observation and output in training_data
                training_data.append([data[0], output])

        env.reset()

        scores.append(score)

        # converting to array
        training_data_save = np.array(training_data)
        # saving training_data
        np.save('training_data_save.npy', training_data_save)

#         average of accepted_Score
    print('mean score', mean(accepted_score))
    print('median score', median(accepted_score))
    print(Counter(accepted_score))

    return training_data


def neural_network(input_size):
    model = Sequential()
    model.add(Dense(128, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.8))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.8))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.8))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.8))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('X_train: ', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)

    if not model:
        model = neural_network(input_size=len(X[0]))

    model.fit(X_train, y_train, epochs=30, validation_data=[X_test, y_test])
    print(model)
    return model


training_data = init_population()
model = train_model(training_data)


scores = []
choice = []
for _ in range(1000):
    score = 0
    game_memory = []
    prev_observation = []
    observation = env.reset()
    for _ in range(200):
        # env.render()
        if len(prev_observation) >= 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_observation))
            print(action)
        choice.append(action)
        new_observation, reward, info, done = env.step(action)
        prev_observation = new_observation
        score += reward
        game_memory.append([prev_observation, scores])
        if done:
            break
    # print(score)
    scores.append(score)

print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choice.count(1) / len(choice), choice.count(0) / len(choice)))
print(score_requirement)



