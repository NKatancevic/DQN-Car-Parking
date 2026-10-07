import math
import csv
import random
from collections import Counter, deque

import numpy as np
import torch

from parking_lot_v2 import ParkingLotV2
from parking_env_final import Car
from advanced_dqn_agent import AdvancedDQNAgent




STATE_SIZE = 31
ACTION_SIZE = 8

TOTAL_EPISODES = 5000

EVAL_EVERY = 100
EVAL_GAMES = 100

SEED_BASE = 51000000




START_PHASE = 3


PHASE_NAMES = {
    0: "TOP6_EMPTY",
    1: "TOP6_OBSTACLES",
    2: "TOP_RANDOM",
    3: "LEFT_RANDOM",
    4: "RIGHT_RANDOM",
    5: "ALL_RANDOM",
}


PHASE_TARGETS = {
    0: [6],
    1: [6],
    2: [6, 7],
    3: [0, 1, 2],
    4: [3, 4, 5],
    5: list(range(8)),
}


PHASE_THRESHOLDS = {
    0: (70.0, 15.0),
    1: (70.0, 15.0),
    2: (65.0, 15.0),
    3: (60.0, 18.0),
    4: (60.0, 18.0),
    5: (70.0, 15.0),
}




BEST_FILES = {
    phase: f"parking_V3_PHASE{phase}_BEST.pth"
    for phase in range(6)
}

LATEST_MODEL = "parking_V3_LATEST.pth"
FINAL_MODEL = "parking_V3_FINAL.pth"

CSV_FILE = "parking_V3_training.csv"



ACTION_NAMES = {
    0: "NISTA",
    1: "NAPRED",
    2: "RIKVERC",
    3: "NAPRED_LEVO",
    4: "NAPRED_DESNO",
    5: "RIKVERC_LEVO",
    6: "RIKVERC_DESNO",
    7: "KOCENJE",
}




random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)



def create_agent():

    return AdvancedDQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,

        lr=0.00001,

        gamma=0.99,

        epsilon=0.40,
        epsilon_min=0.05,
        epsilon_decay=0.998,

        batch_size=256,
        replay_capacity=100000
    )


agent = create_agent()

agent.load(
    "parking_V3_PHASE2_BEST.pth",
    load_optimizer=False
)

agent.epsilon = 0.15

try:

    DEVICE = next(
        agent.model.parameters()
    ).device

except Exception:

    DEVICE = torch.device("cpu")


print("DEVICE:", DEVICE)




def reset_for_phase(
    env,
    phase
):

    env.steps = 0

    env.parking_stage = False
    env.stuck_counter = 0




    env.target_slot_index = random.choice(
        PHASE_TARGETS[
            phase
        ]
    )


    env.parking_rect = (
        env.parking_slots[
            env.target_slot_index
        ]
    )




    env.obstacles = []



    if phase != 0:

        for index, slot in enumerate(
            env.parking_slots
        ):

            if (
                index
                ==
                env.target_slot_index
            ):
                continue


            env.obstacles.append(
                env.create_parked_car(
                    slot
                )
            )




    if phase <= 2:


        start_x = 450.0
        start_y = 525.0
        start_angle = 90.0

    else:


        start_x = (
            450.0
            + random.uniform(
                -10.0,
                10.0
            )
        )

        start_y = (
            525.0
            + random.uniform(
                -8.0,
                8.0
            )
        )

        start_angle = (
            90.0
            + random.uniform(
                -4.0,
                4.0
            )
        )


    env.car = Car(
        start_x,
        start_y
    )


    env.car.angle = (
        start_angle % 360
    )

    env.car.speed = 0.0

    env.car.crashed = False
    env.car.parked = False




    if phase == 0:

        env.max_steps = 500

    elif phase <= 2:

        env.max_steps = 650

    else:

        env.max_steps = 750



    env.previous_nav_distance = (
        env.distance_to_navigation_target()
    )


    geometry = (
        env.get_slot_geometry()
    )


    env.previous_balance_error = (

        abs(
            geometry[
                "horizontal_balance"
            ]
        )

        +

        abs(
            geometry[
                "vertical_balance"
            ]
        )
    )


    env.previous_outside_error = (
        env.get_outside_error(
            geometry
        )
    )


    env.previous_corners_inside = (
        env.corners_inside_parking()
    )


    return env.get_state()




def get_safe_actions(
    env
):

    distances = (
        env.car.get_sensor_readings(
            env.obstacles
        )
    )


    safe = list(
        range(
            ACTION_SIZE
        )
    )




    front_left = distances[1]
    front = distances[2]
    front_right = distances[3]

    rear_right = distances[5]
    rear = distances[6]
    rear_left = distances[7]




    if front < 0.10:

        for action in [
            1,
            3,
            4
        ]:

            if action in safe:

                safe.remove(
                    action
                )


    if (
        front_left < 0.08
        and
        3 in safe
    ):

        safe.remove(3)


    if (
        front_right < 0.08
        and
        4 in safe
    ):

        safe.remove(4)




    if rear < 0.10:

        for action in [
            2,
            5,
            6
        ]:

            if action in safe:

                safe.remove(
                    action
                )


    if (
        rear_left < 0.08
        and
        5 in safe
    ):

        safe.remove(5)


    if (
        rear_right < 0.08
        and
        6 in safe
    ):

        safe.remove(6)




    if not safe:

        return [
            0,
            7
        ]


    return safe




def choose_masked_greedy_action(
    current_agent,
    state,
    safe_actions
):

    tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    )




    if tensor.dim() == 1:

        tensor = (
            tensor.unsqueeze(0)
        )


    current_agent.model.eval()


    with torch.no_grad():

        q_values = (
            current_agent.model(
                tensor
            )
        )



    if q_values.dim() == 2:

        q_values = (
            q_values[0]
        )


    best_action = (
        safe_actions[0]
    )


    best_q = float(
        q_values[
            best_action
        ].item()
    )


    for action in safe_actions[1:]:

        q = float(
            q_values[
                action
            ].item()
        )


        if q > best_q:

            best_q = q
            best_action = action


    return best_action



def teacher_action(
    env
):

    target_x, target_y = (
        env.get_navigation_target()
    )


    dx = (
        target_x
        - env.car.x
    )

    dy = (
        target_y
        - env.car.y
    )


    desired_angle = (
        math.degrees(
            math.atan2(
                -dy,
                dx
            )
        )
        % 360
    )


    current_angle = (
        env.car.angle
        % 360
    )


    angle_error = (

        (
            desired_angle
            - current_angle
            + 180
        )

        % 360

    ) - 180


    distance = math.sqrt(
        dx * dx
        +
        dy * dy
    )


    speed = abs(
        env.car.speed
    )


    safe_actions = (
        get_safe_actions(
            env
        )
    )




    if (
        distance < 60
        and
        speed > 0.8
        and
        7 in safe_actions
    ):

        return 7




    if angle_error > 10:

        if 3 in safe_actions:

            return 3


    if angle_error < -10:

        if 4 in safe_actions:

            return 4




    if 1 in safe_actions:

        return 1




    return random.choice(
        safe_actions
    )




def choose_training_action(
    current_agent,
    env,
    state,
    episode,
    phase
):

    safe_actions = (
        get_safe_actions(
            env
        )
    )




    if phase == 0:

        teacher_start = 0.65

    elif phase == 1:

        teacher_start = 0.45

    elif phase == 2:

        teacher_start = 0.30

    else:

        teacher_start = 0.15


    teacher_probability = max(

        0.0,

        teacher_start
        *
        (
            1.0
            - episode / 1200.0
        )
    )




    if (
        random.random()
        <
        teacher_probability
    ):

        return (
            teacher_action(
                env
            ),
            True
        )




    if (
        random.random()
        <
        current_agent.epsilon
    ):

        return (
            random.choice(
                safe_actions
            ),
            False
        )




    return (
        choose_masked_greedy_action(
            current_agent,
            state,
            safe_actions
        ),
        False
    )




def add_episode_to_replay(
    current_agent,
    transitions,
    outcome
):

    if not transitions:

        return




    if outcome == "SUCCESS":

        for transition in transitions:

            current_agent.remember(
                *transition
            )


        return




    if outcome == "CRASH":

        danger_count = min(
            150,
            len(transitions)
        )


        danger_zone = (
            transitions[
                -danger_count:
            ]
        )



        for transition in danger_zone:

            current_agent.remember(
                *transition
            )


        older = (
            transitions[
                :-danger_count
            ]
        )


        if older:

            amount = min(

                100,

                max(
                    1,
                    int(
                        len(older)
                        * 0.25
                    )
                )
            )


            selected = random.sample(

                older,

                min(
                    amount,
                    len(older)
                )
            )


            for transition in selected:

                current_agent.remember(
                    *transition
                )


        return




    final_count = min(
        80,
        len(transitions)
    )


    final_zone = (
        transitions[
            -final_count:
        ]
    )


    for transition in final_zone:

        current_agent.remember(
            *transition
        )


    older = (
        transitions[
            :-final_count
        ]
    )


    if older:

        amount = max(

            1,

            int(
                len(older)
                * 0.40
            )
        )


        selected = random.sample(

            older,

            min(
                amount,
                len(older)
            )
        )


        for transition in selected:

            current_agent.remember(
                *transition
            )




def get_q_stats(
    states
):

    if not states:

        return {
            "mean_q": 0.0,
            "mean_max_q": 0.0,
            "min_q": 0.0,
            "max_q": 0.0,
        }


    tensor = torch.tensor(

        np.asarray(
            states,
            dtype=np.float32
        ),

        dtype=torch.float32,

        device=DEVICE
    )





    agent.model.eval()


    with torch.no_grad():

        q = (
            agent.model(
                tensor
            )
        )


    q = (
        q.detach()
        .cpu()
        .numpy()
    )


    return {

        "mean_q":
            float(
                np.mean(q)
            ),

        "mean_max_q":
            float(
                np.mean(
                    np.max(
                        q,
                        axis=1
                    )
                )
            ),

        "min_q":
            float(
                np.min(q)
            ),

        "max_q":
            float(
                np.max(q)
            )
    }




def evaluate(
    current_agent,
    phase,
    games=100
):

    old_epsilon = (
        current_agent.epsilon
    )


    current_agent.epsilon = 0.0


    success = 0
    crash = 0
    timeout = 0

    total_steps = 0


    action_counter = (
        Counter()
    )


    diagnostic_states = []


    side_results = {

        "LEFT": {
            "games": 0,
            "success": 0,
            "crash": 0,
            "timeout": 0
        },

        "RIGHT": {
            "games": 0,
            "success": 0,
            "crash": 0,
            "timeout": 0
        },

        "TOP": {
            "games": 0,
            "success": 0,
            "crash": 0,
            "timeout": 0
        }
    }


    for game in range(
        games
    ):

        seed = (
            SEED_BASE
            +
            phase * 10000
            +
            game
        )


        random.seed(seed)
        np.random.seed(seed)


        env = ParkingLotV2(
            start_level=0
        )


        state = reset_for_phase(
            env,
            phase
        )


        side = (
            env.get_target_side()
        )


        side_results[
            side
        ][
            "games"
        ] += 1


        done = False


        while not done:

            if (
                len(
                    diagnostic_states
                )
                < 10000
            ):

                diagnostic_states.append(

                    np.asarray(
                        state,
                        dtype=np.float32
                    )
                )


            safe_actions = (
                get_safe_actions(
                    env
                )
            )


            action = (
                choose_masked_greedy_action(
                    current_agent,
                    state,
                    safe_actions
                )
            )


            action_counter[
                action
            ] += 1


            (
                state,
                reward,
                done
            ) = env.step(
                action
            )


        total_steps += (
            env.steps
        )


        if env.car.parked:

            success += 1

            side_results[
                side
            ][
                "success"
            ] += 1


        elif env.car.crashed:

            crash += 1

            side_results[
                side
            ][
                "crash"
            ] += 1


        else:

            timeout += 1

            side_results[
                side
            ][
                "timeout"
            ] += 1


    current_agent.epsilon = (
        old_epsilon
    )


    q_stats = (
        get_q_stats(
            diagnostic_states
        )
    )


    total_actions = max(

        1,

        sum(
            action_counter.values()
        )
    )


    action_pct = {

        action:

            action_counter[
                action
            ]
            /
            total_actions
            * 100.0

        for action in range(
            ACTION_SIZE
        )
    }


    return {

        "success":
            success,

        "crash":
            crash,

        "timeout":
            timeout,

        "success_rate":
            success
            / games
            * 100.0,

        "crash_rate":
            crash
            / games
            * 100.0,

        "timeout_rate":
            timeout
            / games
            * 100.0,

        "avg_steps":
            total_steps
            / games,

        "actions":
            action_pct,

        "sides":
            side_results,

        "q":
            q_stats
    }




def calculate_score(
    result
):

    return (

        result[
            "success_rate"
        ]

        -

        result[
            "crash_rate"
        ]

        -

        0.30
        *
        result[
            "timeout_rate"
        ]
    )




def print_result(
    result
):

    score = (
        calculate_score(
            result
        )
    )


    print(
        f"SUCCESS: "
        f"{result['success']}/100 "
        f"({result['success_rate']:.1f}%)"
    )


    print(
        f"CRASH: "
        f"{result['crash']}/100 "
        f"({result['crash_rate']:.1f}%)"
    )


    print(
        f"TIMEOUT: "
        f"{result['timeout']}/100 "
        f"({result['timeout_rate']:.1f}%)"
    )


    print(
        f"AVG STEPS: "
        f"{result['avg_steps']:.1f}"
    )


    print(
        f"SCORE: "
        f"{score:.2f}"
    )


    print()
    print("SIDE RESULTS")


    for side in [
        "LEFT",
        "RIGHT",
        "TOP"
    ]:

        data = (
            result[
                "sides"
            ][
                side
            ]
        )


        print(

            f"{side:5s} | "

            f"N={data['games']:3d} | "

            f"S={data['success']:3d} | "

            f"C={data['crash']:3d} | "

            f"T={data['timeout']:3d}"
        )


    print()
    print("Q VALUES")


    print(
        f"MEAN Q:     "
        f"{result['q']['mean_q']:.4f}"
    )


    print(
        f"MEAN MAX Q: "
        f"{result['q']['mean_max_q']:.4f}"
    )


    print(
        f"MIN Q:      "
        f"{result['q']['min_q']:.4f}"
    )


    print(
        f"MAX Q:      "
        f"{result['q']['max_q']:.4f}"
    )


    print()
    print("ACTIONS")


    for action in range(
        ACTION_SIZE
    ):

        print(

            f"{action} "
            f"{ACTION_NAMES[action]:15s}: "

            f"{result['actions'][action]:6.2f}%"
        )


    return score




with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "episode",
        "phase",

        "success",
        "crash",
        "timeout",

        "avg_steps",
        "score",

        "epsilon",
        "avg_loss",

        "mean_q",
        "mean_max_q",
        "min_q",
        "max_q",

        "action0",
        "action1",
        "action2",
        "action3",
        "action4",
        "action5",
        "action6",
        "action7"
    ])




phase = (
    START_PHASE
)


best_score = (
    -999999.0
)


recent_results = deque(
    maxlen=100
)


recent_losses = deque(
    maxlen=300
)


print()
print("=" * 90)

print(
    "PARKING LOT V3 - SNAKE STYLE"
)

print("=" * 90)

print(
    f"START PHASE: "
    f"{phase} "
    f"{PHASE_NAMES[phase]}"
)

print()




for episode in range(
    1,
    TOTAL_EPISODES + 1
):

    env = ParkingLotV2(
        start_level=0
    )


    state = reset_for_phase(
        env,
        phase
    )


    transitions = []

    done = False

    total_reward = 0.0

    teacher_steps = 0



    while not done:

        (
            action,
            used_teacher
        ) = choose_training_action(
            agent,
            env,
            state,
            episode,
            phase
        )


        if used_teacher:

            teacher_steps += 1


        (
            next_state,
            reward,
            done
        ) = env.step(
            action
        )


        transition = (

            state,
            action,
            reward,
            next_state,
            done
        )


        transitions.append(
            transition
        )




        agent.remember(
            *transition
        )



        if (
            env.steps % 4 == 0
        ):

            loss = (
                agent.train_step()
            )


            if loss is not None:

                try:

                    value = float(
                        loss
                    )

                except Exception:

                    try:

                        value = float(
                            loss.item()
                        )

                    except Exception:

                        value = None


                if (
                    value is not None
                    and
                    np.isfinite(
                        value
                    )
                ):

                    recent_losses.append(
                        value
                    )


        state = (
            next_state
        )


        total_reward += (
            reward
        )




    if env.car.parked:

        outcome = "SUCCESS"

    elif env.car.crashed:

        outcome = "CRASH"

    else:

        outcome = "TIMEOUT"


    recent_results.append(
        outcome
    )




    add_episode_to_replay(
        agent,
        transitions,
        outcome
    )



    for _ in range(20):

        loss = (
            agent.train_step()
        )


        if loss is not None:

            try:

                value = float(
                    loss
                )

            except Exception:

                try:

                    value = float(
                        loss.item()
                    )

                except Exception:

                    value = None


            if (
                value is not None
                and
                np.isfinite(
                    value
                )
            ):

                recent_losses.append(
                    value
                )




    if (
        agent.epsilon
        >
        agent.epsilon_min
    ):

        agent.epsilon *= (
            agent.epsilon_decay
        )


        agent.epsilon = max(
            agent.epsilon,
            agent.epsilon_min
        )




    if (
        episode % 50
        == 0
    ):

        agent.update_target_model()




    if (
        episode % 25
        == 0
    ):

        if recent_losses:

            avg_loss = (
                sum(
                    recent_losses
                )
                /
                len(
                    recent_losses
                )
            )

        else:

            avg_loss = float(
                "nan"
            )


        teacher_pct = (

            teacher_steps

            /

            max(
                1,
                env.steps
            )

            * 100.0
        )


        print(

            f"EP {episode:4d}/{TOTAL_EPISODES} | "

            f"P{phase} "
            f"{PHASE_NAMES[phase]:15s} | "

            f"{outcome:7s} | "

            f"target={env.target_slot_index} | "

            f"reward={total_reward:8.2f} | "

            f"eps={agent.epsilon:.3f} | "

            f"teacher={teacher_pct:5.1f}% | "

            f"loss={avg_loss:.4f} | "

            f"last100: "
            f"S={recent_results.count('SUCCESS')} "
            f"C={recent_results.count('CRASH')} "
            f"T={recent_results.count('TIMEOUT')}"
        )




    if (
        episode % EVAL_EVERY
        == 0
    ):

        checkpoint = (

            f"parking_V3_"
            f"P{phase}_"
            f"EP{episode:04d}.pth"
        )


        agent.save(
            checkpoint
        )


        agent.save(
            LATEST_MODEL
        )


        print()
        print("=" * 90)

        print(
            f"EVALUATION EP {episode}"
        )


        print(
            f"PHASE {phase}: "
            f"{PHASE_NAMES[phase]}"
        )


        print(
            f"CHECKPOINT: "
            f"{checkpoint}"
        )

        print()


        result = evaluate(
            agent,
            phase,
            games=EVAL_GAMES
        )


        score = (
            print_result(
                result
            )
        )




        if recent_losses:

            avg_loss = (
                sum(
                    recent_losses
                )
                /
                len(
                    recent_losses
                )
            )

        else:

            avg_loss = float(
                "nan"
            )



        if (
            score
            >
            best_score
        ):

            best_score = (
                score
            )


            agent.save(
                BEST_FILES[
                    phase
                ]
            )


            print()
            print(
                f">>> NEW PHASE "
                f"{phase} BEST <<<"
            )




        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow([

                episode,
                phase,

                result[
                    "success_rate"
                ],

                result[
                    "crash_rate"
                ],

                result[
                    "timeout_rate"
                ],

                result[
                    "avg_steps"
                ],

                score,

                agent.epsilon,

                avg_loss,

                result[
                    "q"
                ][
                    "mean_q"
                ],

                result[
                    "q"
                ][
                    "mean_max_q"
                ],

                result[
                    "q"
                ][
                    "min_q"
                ],

                result[
                    "q"
                ][
                    "max_q"
                ],

                result[
                    "actions"
                ][0],

                result[
                    "actions"
                ][1],

                result[
                    "actions"
                ][2],

                result[
                    "actions"
                ][3],

                result[
                    "actions"
                ][4],

                result[
                    "actions"
                ][5],

                result[
                    "actions"
                ][6],

                result[
                    "actions"
                ][7],
            ])




        (
            required_success,
            max_crash
        ) = (
            PHASE_THRESHOLDS[
                phase
            ]
        )


        if (

            result[
                "success_rate"
            ]
            >= required_success

            and

            result[
                "crash_rate"
            ]
            <= max_crash
        ):

            print()
            print(
                "########################################"
            )

            print(
                f"PHASE {phase} MASTERED"
            )

            print(
                f"SUCCESS: "
                f"{result['success_rate']:.1f}%"
            )

            print(
                f"CRASH: "
                f"{result['crash_rate']:.1f}%"
            )

            print(
                "########################################"
            )


            agent.save(
                BEST_FILES[
                    phase
                ]
            )




            if phase == 5:

                agent.save(
                    FINAL_MODEL
                )


                print()
                print(
                    "ALL PHASES MASTERED."
                )

                break




            phase += 1


            print()
            print(
                f">>> MOVING TO PHASE "
                f"{phase}: "
                f"{PHASE_NAMES[phase]}"
            )



            if phase <= 2:

                agent.epsilon = 0.25

            else:

                agent.epsilon = 0.15



            recent_results.clear()
            recent_losses.clear()

            best_score = -999999.0



        max_action = max(

            result[
                "actions"
            ],

            key=result[
                "actions"
            ].get
        )


        if (

            result[
                "actions"
            ][
                max_action
            ]
            > 65.0
        ):

            print()
            print(
                "WARNING: POSSIBLE POLICY COLLAPSE"
            )


            print(
                "Dominant action:",
                ACTION_NAMES[
                    max_action
                ],
                f"{result['actions'][max_action]:.1f}%"
            )




        if (
            abs(
                result[
                    "q"
                ][
                    "max_q"
                ]
            )
            > 100.0
        ):

            print()
            print(
                "WARNING: Q VALUES TOO LARGE"
            )


        print("=" * 90)
        print()




agent.save(
    LATEST_MODEL
)


print()
print("=" * 90)

print(
    "TRAINING STOPPED / FINISHED"
)

print(
    "LATEST:",
    LATEST_MODEL
)

print(
    "CSV:",
    CSV_FILE
)

print("=" * 90)
