import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingDQN(nn.Module):

    def __init__(
        self,
        state_size,
        action_size
    ):

        super().__init__()



        self.feature = nn.Sequential(

            nn.Linear(
                state_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU()
        )




        self.value_stream = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )




        self.advantage_stream = nn.Sequential(

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                action_size
            )
        )


    def forward(
        self,
        x
    ):

        features = self.feature(
            x
        )

        value = self.value_stream(
            features
        )

        advantage = self.advantage_stream(
            features
        )




        q_values = (
            value
            +
            advantage
            -
            advantage.mean(
                dim=1,
                keepdim=True
            )
        )


        return q_values




class ReplayBuffer:

    def __init__(
        self,
        capacity=100000
    ):

        self.capacity = capacity

        self.buffer = deque(
            maxlen=capacity
        )


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                np.array(
                    state,
                    dtype=np.float32
                ),

                int(action),

                float(reward),

                np.array(
                    next_state,
                    dtype=np.float32
                ),

                bool(done)
            )
        )


    def sample(
        self,
        batch_size
    ):

        batch = random.sample(
            self.buffer,
            batch_size
        )


        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(
            *batch
        )


        return (

            np.array(
                states,
                dtype=np.float32
            ),

            np.array(
                actions,
                dtype=np.int64
            ),

            np.array(
                rewards,
                dtype=np.float32
            ),

            np.array(
                next_states,
                dtype=np.float32
            ),

            np.array(
                dones,
                dtype=np.float32
            )
        )


    def __len__(
        self
    ):

        return len(
            self.buffer
        )




class AdvancedDQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        lr=0.0002,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.03,
        epsilon_decay=0.997,
        batch_size=128,
        replay_capacity=100000
    ):

        self.state_size = state_size
        self.action_size = action_size




        self.gamma = gamma

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = (
            epsilon_decay
        )

        self.batch_size = (
            batch_size
        )

        self.lr = lr




        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )


        print(
            "Advanced DQN device:",
            self.device
        )



        self.model = DuelingDQN(
            state_size,
            action_size
        ).to(
            self.device
        )



        self.target_model = DuelingDQN(
            state_size,
            action_size
        ).to(
            self.device
        )



        self.update_target_model()




        self.optimizer = (
            optim.Adam(
                self.model.parameters(),
                lr=self.lr
            )
        )




        self.loss_function = (
            nn.SmoothL1Loss()
        )



        self.memory = ReplayBuffer(
            replay_capacity
        )




        self.training_steps = 0




    def choose_action(
        self,
        state,
        greedy=False
    ):



        if (
            not greedy
            and
            random.random()
            < self.epsilon
        ):

            return random.randrange(
                self.action_size
            )




        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(
            0
        )


        self.model.eval()


        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )


        self.model.train()


        return int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )




    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )




    def train_step(
        self
    ):

        if (
            len(self.memory)
            < self.batch_size
        ):

            return None


        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )


        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )


        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(
            1
        )


        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(
            1
        )


        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )


        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(
            1
        )




        current_q = self.model(
            states
        ).gather(
            1,
            actions
        )



        with torch.no_grad():

            next_actions = (
                self.model(
                    next_states
                )
                .argmax(
                    dim=1,
                    keepdim=True
                )
            )


            next_q = (
                self.target_model(
                    next_states
                )
                .gather(
                    1,
                    next_actions
                )
            )


            target_q = (
                rewards
                +
                self.gamma
                * next_q
                * (
                    1.0
                    - dones
                )
            )




        loss = self.loss_function(
            current_q,
            target_q
        )



        self.optimizer.zero_grad()

        loss.backward()




        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )


        self.optimizer.step()


        self.training_steps += 1


        return float(
            loss.item()
        )




    def update_target_model(
        self
    ):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()



    def decay_epsilon(
        self
    ):

        if (
            self.epsilon
            > self.epsilon_min
        ):

            self.epsilon *= (
                self.epsilon_decay
            )


            if (
                self.epsilon
                < self.epsilon_min
            ):

                self.epsilon = (
                    self.epsilon_min
                )


    def set_epsilon(
        self,
        epsilon
    ):

        self.epsilon = max(
            self.epsilon_min,
            float(epsilon)
        )




    def set_learning_rate(
        self,
        learning_rate
    ):

        self.lr = learning_rate


        for param_group in (
            self.optimizer.param_groups
        ):

            param_group["lr"] = (
                learning_rate
            )




    def save(
        self,
        path
    ):

        torch.save(

            {
                "model": (
                    self.model.state_dict()
                ),

                "target_model": (
                    self.target_model.state_dict()
                ),

                "optimizer": (
                    self.optimizer.state_dict()
                ),

                "epsilon": (
                    self.epsilon
                ),

                "training_steps": (
                    self.training_steps
                )
            },

            path
        )




    def load(
        self,
        path,
        load_optimizer=True
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )




        if (
            isinstance(
                checkpoint,
                dict
            )

            and

            "model"
            in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model"]
            )


            if (
                "target_model"
                in checkpoint
            ):

                self.target_model.load_state_dict(
                    checkpoint[
                        "target_model"
                    ]
                )

            else:

                self.update_target_model()


            if (
                load_optimizer
                and
                "optimizer"
                in checkpoint
            ):

                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer"
                    ]
                )


            if (
                "epsilon"
                in checkpoint
            ):

                self.epsilon = float(
                    checkpoint["epsilon"]
                )


            if (
                "training_steps"
                in checkpoint
            ):

                self.training_steps = int(
                    checkpoint[
                        "training_steps"
                    ]
                )




        else:

            self.model.load_state_dict(
                checkpoint
            )

            self.update_target_model()


        self.model.to(
            self.device
        )

        self.target_model.to(
            self.device
        )
