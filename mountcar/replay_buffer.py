import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self,capacity):
     self.buffer =deque(maxlen=capacity)#maxlen=capacityは古い要素から自動で消える仕組み

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))#情報を追加

    def sample(self,batch_size):
        batch =random.sample(self.buffer,batch_size)#経験を選ぶ
        states,actions,rewards,next_state,dones = map(np.array,zip(*batch))#zipは列ごとに取り出す,mapは全要素
        return states,actions,rewards,next_state,dones
     
    def __len__(self):
        return len(self.buffer)#経験の数を返す