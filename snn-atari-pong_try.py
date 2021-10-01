

import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env



class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)
    

def wrap_pytorch(env):
    return ImageToPyTorch(env)

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

class LinearSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the piecewise-linear surrogate gradient as was
        done in Bellec et al. (2018).
        """
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = grad_input*LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad

spike_fn = LinearSpike.apply  


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3136, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, self.num_actions, bias=False)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

class SnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1=nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, bias=False)
        self.conv2=nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3=nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        self.fc1=nn.Linear(3136, 512, bias=False)
        self.head=nn.Linear(512, self.num_actions, bias=False)   
        
        #90p th- obtained during DQN-SNN conversion
        
#        self.conv1.threshold=10.058
#        self.conv2.threshold=.3668
#        self.conv3.threshold=.1975
#        self.fc1.threshold=0.1878
        
        threshold 	= {}
        
        threshold['t0'] 	= nn.Parameter(torch.tensor(1.0))
        threshold['t1'] 	= nn.Parameter(torch.tensor(1.0))
        threshold['t2'] 	= nn.Parameter(torch.tensor(1.0))
        threshold['t3'] 	= nn.Parameter(torch.tensor(1.0))
        
        self.threshold 	= nn.ParameterDict(threshold)
        

        
    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        self.scaling_factor = scaling_factor
        self.threshold.update({'t0': nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
        self.threshold.update({'t1': nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
        self.threshold.update({'t2': nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
        self.threshold.update({'t3': nn.Parameter(torch.tensor(thresholds.pop(0))*self.scaling_factor)})
        
        
        

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result   
    
    def forward(self, x):
        batch_size=x.size(0)
        
        self.mem_conv1 = self.spike_conv1 = torch.zeros(batch_size, 32, 20, 20).cuda()
        self.mem_conv2 = self.spike_conv2 = torch.zeros(batch_size, 64, 9, 9).cuda()
        self.mem_conv3 = self.spike_conv3 = torch.zeros(batch_size, 64, 7, 7).cuda()

        self.mem_fc1  = self.spike_fc1 = torch.zeros(batch_size, 512).cuda()
          
        self.mem_head  = torch.zeros(batch_size, self.num_actions).cuda()
        
        

        for t in range(1):
          
          self.mem_conv1 = self.mem_conv1 + self.conv1(x)
          
        
          
          mem_thr        = (self.mem_conv1/getattr(self.threshold, 't0')) - 1.0
          self.spike_conv1       = spike_fn(mem_thr)

          rst=getattr(self.threshold, 't0')*(mem_thr>0).float()
          self.mem_conv1 = self.mem_conv1-rst
          
         
#
#          #print(self.conv2(self.spike_conv1).shape)
#
          self.mem_conv2 = self.mem_conv2 + self.conv2(self.spike_conv1)
          
          mem_thr        = (self.mem_conv2/getattr(self.threshold, 't1')) - 1.0
          self.spike_conv2       = spike_fn(mem_thr)

          rst=getattr(self.threshold, 't1')*(mem_thr>0).float()
          self.mem_conv2 = self.mem_conv2-rst
          
          
#
#
          self.mem_conv3 = self.mem_conv3 + self.conv3(self.spike_conv2)
          mem_thr        = (self.mem_conv3/getattr(self.threshold, 't2')) - 1.0
          self.spike_conv3       = spike_fn(mem_thr)

          rst=getattr(self.threshold, 't2')*(mem_thr>0).float()
          self.mem_conv3 = self.mem_conv3-rst

          out=self.spike_conv3.reshape(batch_size, -1)
          
     
#
#
          self.mem_fc1 = self.mem_fc1 + self.fc1(out)
          mem_thr        = (self.mem_fc1/getattr(self.threshold, 't3')) - 1.0
          self.spike_fc1       = spike_fn(mem_thr)

          rst=getattr(self.threshold, 't3')*(mem_thr>0).float()
          self.mem_fc1= self.mem_fc1-rst

          self.mem_head  =self.mem_head  + self.head (self.spike_fc1)
        

       
        return self.mem_head
    
    #def feature_size(self):
     #   return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action
    
model = SnnDQN(env.observation_space.shape, env.action_space.n)

#model = CnnDQN(env.observation_space.shape, env.action_space.n)


#pretrained_state    = './ann_atari_initial.pth'
pretrained_state    = './snn_atari_tst3_diet_noframestack.pth'


weights = torch.load(pretrained_state, map_location='cpu')  
missing_keys, unexpected_keys = model.load_state_dict(weights['model_state_dict'], strict=False)
print('\n Missing keys : {}, Unexpected Keys: {}'.format(missing_keys, unexpected_keys))    

# Vth from previously trained model (T3)
#thresholds =[9.7313, .163, .2393, 0.6011]

thresholds =[model.threshold.t0.item(),model.threshold.t1.item(),model.threshold.t2.item(),model.threshold.t3.item()]


#        thresholds = find_threshold(batch_size=512, timesteps=2000, architecture=architecture)
model.threshold_update(scaling_factor = 1.0, thresholds=thresholds[:])


if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('ANN-DQN frame %s. Mean Recent Reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    #plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    
# snn


batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0


#if resume training, then load previous rewards and losses
#c=np.load("atari-ann-rewards.npz")
#d=c['arr_0']
#d=d.tolist()
#all_rewards.extend(d)
#
#e=np.load("atari-ann-loss.npz", allow_pickle=True)
#f=e['arr_0']
#f=f.tolist()
#losses.extend(f)

#c=np.load("atari-hybrid-tst5-initial-rewards.npz")
#d=c['arr_0']
#d=d.tolist()
#all_rewards.extend(d)
#
#e=np.load("atari-hybrid-tst5-initial-loss.npz", allow_pickle=True)
#f=e['arr_0']
#f=f.tolist()
#losses.extend(f)

state = env.reset()

#num_frames = 1800000

num_frames = 2500000
for frame_idx in range(1, num_frames + 1):
    #print(frame_idx)
    #epsilon = epsilon_by_frame(frame_idx)
    epsilon = .01
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.data)
        
    if frame_idx % 10000 == 0:
        plot(frame_idx, all_rewards, losses)
        print(model.threshold.items())

ckpt = {'model_state_dict': model.state_dict(),
               'optim_state_dict': optimizer.state_dict()}
#ckpt_fname= 'ann_atari_dqn_snn_tst10.pth'
ckpt_fname= 'snn_atari_tst1_diet.pth'
torch.save(ckpt, ckpt_fname)

