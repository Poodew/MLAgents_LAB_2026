import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel

# 파라미터 값 세팅
state_size = 4
action_size = 5

load_model = False
train_mode = True

discount_factor = 0.9
learning_rate = 0.00025

max_episodes = 1000 #최대 Episode
max_step = 500 #한 에피소드 당 최대 step
test_start_episode = 400 #신경망 학습(Gradient Update)을 종료하는 Episode

save_interval = 10

#Unity 환경 관련 Parameters
game = "Cartpole" #빌드 파일 이름
os_name = platform.system() #실행시키는 디바이스의 OS
if os_name == "Windows":
    env_name = f"" #빌드 파일 경로

#모델 및 텐서보드 관련 Parameters
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"./save_model" #모델 및 텐서보드 파일 저장 경로
load_path = f"" #모델을 불러올 경로

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A2C 클래스 -> Actor Network, Critic Network 정의
class A2C(torch.nn.Module):
    def __init__(self, **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, 128) #Actor Network, Critic Network의 공용 입력층
        self.d2 = torch.nn.Linear(128, 128)
        self.pi = torch.nn.Linear(128, action_size) #Actor Network Output
        self.v = torch.nn.Linear(128, 1) #Critic Network Output

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return F.softmax(self.pi(x), dim=1), self.v(x)


# A2CAgent 클래스 -> A2C 알고리즘을 위한 다양한 함수 정의
class A2CAgent:
    def __init__(self):
        self.a2c = A2C().to(device)
        self.optimizer = torch.optim.Adam(self.a2c.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.a2c.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 정책을 통해 행동 결정
    def get_action(self, state, training=True):
        self.a2c.train(training)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)  # [4] → [1, 4] 텐서 변환

        with torch.no_grad():
            pi, _ = self.a2c(state_tensor)

        if training:
            action = torch.multinomial(pi, num_samples=1)
        else:
            action = torch.argmax(pi, dim=1, keepdim=True)

        return action.item()

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        #학습 전 트랜지션(s, a, r, s', done) 텐서화
        state, action, reward, next_state, done = (
            self.to_transition_tensor(state, action, reward, next_state, done))
        pi, value = self.a2c(state)

        # Critic Loss 계산
        with torch.no_grad():
            _, next_value = self.a2c(next_state)
            target_value = reward + (1.0 - done) * discount_factor * next_value #One-step TD Target
        critic_loss = F.mse_loss(value, target_value)

        # Actor Loss 계산
        selected_pi = pi.gather(1, action)
        advantage = (target_value - value).detach()
        actor_loss = -(torch.log(selected_pi.clamp_min(1e-8))* advantage).mean()

        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    #텐서화 함수
    def to_transition_tensor(self, state, action, reward, next_state, done):
        state = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).view(1, state_size) #[1, 4]

        next_state = torch.as_tensor(
            next_state, dtype=torch.float32, device=device
        ).view(1, state_size) #[1, 4]

        action = torch.as_tensor(
            action, dtype=torch.long, device=device
        ).view(1, 1) #[1, 1]

        reward = torch.as_tensor(
            reward, dtype=torch.float32, device=device
        ).view(1, 1) #[1, 1]

        done = torch.as_tensor(
            done, dtype=torch.float32, device=device
        ).view(1, 1) #[1, 1]

        return state, action, reward, next_state, done

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.a2c.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

        # 학습 기록

    def write_summary(self, score, actor_loss, critic_loss, episode):
        self.writer.add_scalar("run/score", score, episode)
        self.writer.add_scalar("model/actor_loss", actor_loss, episode)
        self.writer.add_scalar("model/critic_loss", critic_loss, episode)


# Main 함수 -> 전체적으로 A2C 알고리즘을 진행
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=12.0)
    dec, term = env.get_steps(behavior_name)

    # A2C 클래스를 agent로 정의
    agent = A2CAgent()
    actor_losses, critic_losses, score = [], [], 0
    for episode in range(max_episodes):
        if episode == test_start_episode:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        env.reset()
        dec, term = env.get_steps(behavior_name)

        done = False
        for step in range(max_step):
            state = dec.obs[0][0]
            action = agent.get_action(state, train_mode)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]], dtype=np.int32))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # 환경으로부터 얻는 정보
            dec, term = env.get_steps(behavior_name)
            done = len(term.agent_id) > 0
            reward = float(term.reward[0] if done else dec.reward[0])
            next_state = term.obs[0][0] if done else dec.obs[0][0]
            score += reward

            if train_mode:
                # 학습수행
                actor_loss, critic_loss = agent.train_model(state, action, reward, next_state, done)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            if done or step == max_step - 1:
                break

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
        mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
        mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
        agent.write_summary(score, mean_actor_loss, mean_critic_loss, episode)

        print(f"{episode} Episode / Score: {score:.2f} / " + \
              f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")
        actor_losses, critic_losses, score= [], [] ,0

        # 네트워크 모델 저장
        if train_mode and episode % save_interval == 0:
            agent.save_model()

    env.close()
    agent.writer.close()
