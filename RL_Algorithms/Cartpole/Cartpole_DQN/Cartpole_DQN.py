import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig

'''Parameters for DQN'''

#상태, 행동 크기 셋팅
state_size = 4 #Cart Location, Cart Velocity, Pole Angle, Pole Angle Velocity
action_size = 2 #Left, Right

#학습 방법 설정
load_model = False #이전에 학습한 모델 사용 여부
train_mode = True #학습 여부

#DQN 관련 Parameters
batch_size = 32 #mini-batch 크기
mem_maxlen = 10000 #Replay Memory의 용량
discount_factor = 0.9 #감가율
learning_rate = 0.00025 #학습률

#Episode 관련 Parameters
max_episodes = 1000 #최대 Episode
max_step = 500 #한 에피소드 당 최대 step
test_start_episode = 100 #Replay Memory를 채운 뒤 신경망 학습(Gradient Update)을 시작하는 Episode

#상황 저장 관련 Parameters
save_interval = 10 #모델 저장 Episode 주기

#epsilon 관련 Parameters
epsilon_train = 1.0 # 학습 시작 시 e 확률
epsilon = epsilon_train
epsilon_min = 0.01 # 최소 e 확률
epsilon_decay = 0.995 # e 확률 감소량

#Unity 환경 관련 Parameters
game = "Cartpole" #빌드 파일 이름
os_name = platform.system() #실행시키는 디바이스의 OS
if os_name == "Windows":
    env_name = f"" #빌드 파일 경로

#모델 및 텐서보드 관련 Parameters
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"./save_model" #모델 및 텐서보드 파일 저장 경로
load_path = f"" #모델을 불러올 경로

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #연산 장치 설정

'''Network for DQN'''

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 64) #입력 노드가 4개인 입력층
        self.fc2 = torch.nn.Linear(64, 64) #은닉층
        self.fc3 = torch.nn.Linear(64, action_size) #출력 노드가 2개인 출력층

    def forward(self, x):
        x = F.relu(self.fc1(x)) #입력층 - 은닉층) ReLU 방식으로 결정
        x = F.relu(self.fc2(x)) #은닉층 - 출력층) ReLU 방식으로 결정
        x = self.fc3(x) #출력층 - 최종 출력 Linear 방식으로 결정

        return x

'''DQN Training'''

class DQNAgent:
    def __init__(self):
        self.network = DQN(state_size, action_size).to(device) #Q-Network
        self.target_network = copy.deepcopy(self.network) #Target-Q-Network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate) #가중치 수정
        self.memory = deque(maxlen=mem_maxlen) #리플레이 메모리
        self.epsilon = epsilon_train #epsilon 확률 초기화(1.0)
        self.writer = SummaryWriter(log_dir=save_path) #Tensorboard 기록 저장

    #행동 결정
    def get_action(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_min

        if training and epsilon > random.random(): #epsilon 랜덤 행동 결정
            return np.random.randint(0, action_size)

        with torch.no_grad(): #가장 높은 큐값을 가진 행동으로 결정
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q = self.network(state)
            return torch.argmax(q).item()

    #Replay Memory에 S, A, R, S' 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #DQN 알고리즘을 통한 모델 학습
    def train_model(self):
        batch = random.sample(self.memory, batch_size) #Replay Memory에서 랜덤으로 S, A, R, S' 가져옴
        state = np.stack([b[0] for b in batch], axis = 0)
        action = np.stack([b[1] for b in batch], axis = 0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)

        #S, A, R, S'를 FloatTensor로 변환
        state = torch.FloatTensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).view(-1, 1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).view(-1, 1).to(device)

        #실제로 한 행동의 큐함수값만 추출
        q = self.network(state).gather(1, action.view(-1, 1).long())

        #Target 계산
        with torch.no_grad():
            next_q = self.target_network(next_state) #target_network를 통해 다음 상태의 큐함수값 저장
            target_q = reward + next_q.max(1, keepdim=True).values * ((1 - done) * discount_factor) #Q-Learning

        loss = F.smooth_l1_loss(q, target_q) #q와 target_q값 간의 Huber Loss 계산

        self.optimizer.zero_grad() #Gradient 벡터 초기화
        loss.backward() #역전파 계산을 통한 Gradient 벡터 계산
        self.optimizer.step() #계산된 Gradient 벡터를 이용해 파라미터 업데이트

        return loss.item()

    #Target Network 가중치 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    #네트워크 모델 저장
    def save_model(self):
        print(f"...Save Model to {save_path}/ckpt ...")
        torch.save({"network" : self.network.state_dict(),
                    "optimizer" : self.optimizer.state_dict()}
                   , save_path + "/ckpt.")

    # Tensorboard에 학습 기록
    def write_summary(self, score, loss,  epsilon, episode):
        self.writer.add_scalar("run/score", score, episode)
        self.writer.add_scalar("model/loss", loss, episode)
        self.writer.add_scalar("model/epsilon", epsilon, episode)

if __name__ == '__main__':
    # 유니티 환경 설정(Editor Ver)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    # DQNAgent 클래스를 agent로 정의
    agent = DQNAgent()

    losses, score = [], 0

    for episode in range(max_episodes):

        if episode == test_start_episode:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        done = False
        for step in range(max_step):
            state = dec.obs[0][0]
            action = agent.get_action(state, train_mode)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            dec, term = env.get_steps(behavior_name)
            done = len(term.agent_id) > 0

            reward = term.reward[0] if done else dec.reward[0]
            next_state = term.obs[0][0] if done else dec.obs[0][0]

            score += reward

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)

            if train_mode and len(agent.memory) >= batch_size:
                loss = agent.train_model()
                losses.append(loss)

            if done:
                epsilon *= epsilon_decay  # epsilon 확률 감소
                epsilon = max(epsilon_min, epsilon)
                break

        if train_mode:
            agent.update_target()

        mean_loss = np.mean(losses) if losses else 0
        agent.write_summary(score, mean_loss, agent.epsilon, episode)

        print(f"{episode} Episode / Score: {score:.2f} / "
              f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

        losses, score = [], 0

        if train_mode and episode % save_interval == 0:
            agent.save_model()

    env.close()

















