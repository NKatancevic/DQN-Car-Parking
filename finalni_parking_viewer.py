import math
import random
import pygame
import numpy as np
import torch

from parking_lot_v2 import ParkingLotV2
from parking_env_final import Car, WIDTH, HEIGHT
from advanced_dqn_agent import AdvancedDQNAgent


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "parking_V3_FINAL.pth"

STATE_SIZE = 31
ACTION_SIZE = 8

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 700

PARKING_X = 20
PARKING_Y = 60

FPS = 60

SEED_BASE = 88000000


ACTION_NAMES = {
    0: "NISTA",
    1: "NAPRED",
    2: "RIKVERC",
    3: "NAPRED + LEVO",
    4: "NAPRED + DESNO",
    5: "RIKVERC + LEVO",
    6: "RIKVERC + DESNO",
    7: "KOCENJE",
}


# ============================================================
# COLORS
# ============================================================

BACKGROUND = (15, 18, 22)

ASPHALT = (42, 46, 51)
ASPHALT_LIGHT = (48, 52, 57)

WHITE = (235, 238, 240)
GRAY = (150, 155, 160)
DARK_GRAY = (75, 80, 85)

YELLOW = (245, 190, 55)
GREEN = (70, 220, 120)
RED = (235, 80, 80)
BLUE = (65, 145, 255)
CYAN = (70, 220, 240)

TARGET_FILL = (40, 120, 70)
TARGET_BORDER = (90, 255, 150)

CAR_BODY = (30, 120, 230)
CAR_DARK = (20, 70, 140)

PARKED_BODY = (120, 125, 135)
PARKED_DARK = (70, 75, 85)

WINDOW_COLOR = (45, 65, 85)
LIGHT_COLOR = (255, 245, 190)

PANEL = (22, 26, 31)
PANEL_2 = (30, 35, 41)

SUCCESS_COLOR = (70, 225, 120)
CRASH_COLOR = (235, 75, 75)
TIMEOUT_COLOR = (245, 190, 55)


# ============================================================
# PYGAME
# ============================================================

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode(
    (
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )
)

pygame.display.set_caption(
    "Autonomous Parking - DQN Final Demo"
)

clock = pygame.time.Clock()


font_small = pygame.font.SysFont(
    "segoeui",
    17
)

font_medium = pygame.font.SysFont(
    "segoeui",
    21
)

font_big = pygame.font.SysFont(
    "segoeui",
    30,
    bold=True
)

font_title = pygame.font.SysFont(
    "segoeui",
    34,
    bold=True
)


# ============================================================
# AGENT
# ============================================================

agent = AdvancedDQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,

    lr=0.00001,

    gamma=0.99,

    epsilon=0.0,
    epsilon_min=0.0,
    epsilon_decay=1.0,

    batch_size=256,
    replay_capacity=100000
)


agent.load(
    MODEL_FILE,
    load_optimizer=False
)

agent.epsilon = 0.0


DEVICE = next(
    agent.model.parameters()
).device


# ============================================================
# DRAW HELPERS
# ============================================================

def world_to_screen(
    x,
    y
):

    return (
        int(
            PARKING_X + x
        ),
        int(
            PARKING_Y + y
        )
    )


def draw_text(
    surface,
    text,
    x,
    y,
    font,
    color=WHITE
):

    rendered = font.render(
        str(text),
        True,
        color
    )

    surface.blit(
        rendered,
        (
            x,
            y
        )
    )


# ============================================================
# RESET TACNO KAO FINALNA EVALUACIJA
# ============================================================

def reset_for_slot(
    env,
    target_slot,
    seed
):

    random.seed(seed)
    np.random.seed(seed)


    env.steps = 0

    env.parking_stage = False
    env.stuck_counter = 0


    # ========================================================
    # TARGET
    # ========================================================

    env.target_slot_index = (
        target_slot
    )


    env.parking_rect = (
        env.parking_slots[
            target_slot
        ]
    )


    # ========================================================
    # OSTALA MESTA ZAUZETA
    # ========================================================

    env.obstacles = []


    for index, slot in enumerate(
        env.parking_slots
    ):

        if index == target_slot:
            continue


        env.obstacles.append(
            env.create_parked_car(
                slot
            )
        )


    # ========================================================
    # FINAL PHASE RANDOM START
    # ========================================================

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


    env.max_steps = 750


    # ========================================================
    # HISTORY
    # ========================================================

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


# ============================================================
# SAFE ACTION MASK
# ============================================================

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


    # ========================================================
    # FRONT
    # ========================================================

    if front < 0.10:

        for action in [
            1,
            3,
            4
        ]:

            if action in safe:
                safe.remove(action)


    if (
        front_left < 0.08
        and 3 in safe
    ):

        safe.remove(3)


    if (
        front_right < 0.08
        and 4 in safe
    ):

        safe.remove(4)


    # ========================================================
    # REAR
    # ========================================================

    if rear < 0.10:

        for action in [
            2,
            5,
            6
        ]:

            if action in safe:
                safe.remove(action)


    if (
        rear_left < 0.08
        and 5 in safe
    ):

        safe.remove(5)


    if (
        rear_right < 0.08
        and 6 in safe
    ):

        safe.remove(6)


    if not safe:

        return [
            0,
            7
        ]


    return safe


# ============================================================
# GREEDY DQN ACTION
# ============================================================

def choose_action(
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


    agent.model.eval()


    with torch.no_grad():

        q_values = (
            agent.model(
                tensor
            )
        )[0]


    best_action = safe_actions[0]

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


    return (
        best_action,
        q_values.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# PARKING LOT DRAW
# ============================================================

def draw_parking_background(
    surface
):

    # ========================================================
    # OUTER SHADOW
    # ========================================================

    shadow_rect = pygame.Rect(
        PARKING_X + 8,
        PARKING_Y + 8,
        WIDTH,
        HEIGHT
    )


    pygame.draw.rect(
        surface,
        (5, 7, 9),
        shadow_rect,
        border_radius=12
    )


    # ========================================================
    # ASPHALT
    # ========================================================

    lot_rect = pygame.Rect(
        PARKING_X,
        PARKING_Y,
        WIDTH,
        HEIGHT
    )


    pygame.draw.rect(
        surface,
        ASPHALT,
        lot_rect,
        border_radius=12
    )


    # ========================================================
    # SUBTLE ASPHALT TEXTURE
    # ========================================================

    rng = random.Random(1337)


    for _ in range(600):

        x = (
            PARKING_X
            + rng.randint(
                5,
                WIDTH - 5
            )
        )

        y = (
            PARKING_Y
            + rng.randint(
                5,
                HEIGHT - 5
            )
        )


        shade = rng.choice([
            45,
            48,
            50,
            38
        ])


        pygame.draw.circle(
            surface,
            (
                shade,
                shade,
                shade + 2
            ),
            (
                x,
                y
            ),
            1
        )


    # ========================================================
    # CURB
    # ========================================================

    pygame.draw.rect(
        surface,
        (170, 175, 180),
        lot_rect,
        width=4,
        border_radius=12
    )


    pygame.draw.rect(
        surface,
        (75, 80, 85),
        lot_rect.inflate(
            -10,
            -10
        ),
        width=2,
        border_radius=8
    )


    # ========================================================
    # CENTER DRIVE LANE
    # ========================================================

    lane_x = (
        PARKING_X
        + WIDTH // 2
    )


    for y in range(
        PARKING_Y + 90,
        PARKING_Y + HEIGHT - 70,
        50
    ):

        pygame.draw.line(
            surface,
            (110, 115, 105),
            (
                lane_x,
                y
            ),
            (
                lane_x,
                y + 24
            ),
            3
        )


    # ========================================================
    # ENTRY ARROW
    # ========================================================

    arrow_x = (
        PARKING_X
        + WIDTH // 2
    )

    arrow_y = (
        PARKING_Y
        + HEIGHT - 42
    )


    pygame.draw.line(
        surface,
        (180, 180, 170),
        (
            arrow_x,
            arrow_y
        ),
        (
            arrow_x,
            arrow_y - 35
        ),
        4
    )


    pygame.draw.polygon(
        surface,
        (180, 180, 170),
        [
            (
                arrow_x,
                arrow_y - 48
            ),
            (
                arrow_x - 10,
                arrow_y - 31
            ),
            (
                arrow_x + 10,
                arrow_y - 31
            )
        ]
    )


# ============================================================
# PARKING SLOTS
# ============================================================

def draw_slots(
    surface,
    env
):

    for index, slot in enumerate(
        env.parking_slots
    ):

        rect = pygame.Rect(
            PARKING_X + slot.x,
            PARKING_Y + slot.y,
            slot.width,
            slot.height
        )


        # ====================================================
        # TARGET SLOT
        # ====================================================

        if (
            index
            ==
            env.target_slot_index
        ):

            glow = rect.inflate(
                12,
                12
            )


            target_surface = pygame.Surface(
                (
                    glow.width,
                    glow.height
                ),
                pygame.SRCALPHA
            )


            target_surface.fill(
                (
                    30,
                    200,
                    90,
                    35
                )
            )


            surface.blit(
                target_surface,
                glow.topleft
            )


            pygame.draw.rect(
                surface,
                TARGET_BORDER,
                rect,
                width=4,
                border_radius=4
            )


            draw_text(
                surface,
                "TARGET",
                rect.centerx - 31,
                rect.centery - 10,
                font_medium,
                TARGET_BORDER
            )


        else:

            pygame.draw.rect(
                surface,
                (
                    205,
                    205,
                    190
                ),
                rect,
                width=3
            )


        # slot number
        draw_text(
            surface,
            str(
                index + 1
            ),
            rect.x + 6,
            rect.y + 4,
            font_small,
            (
                190,
                190,
                180
            )
        )


# ============================================================
# DRAW REALISTIC CAR
# ============================================================

def create_car_surface(
    body_color,
    dark_color,
    player=False
):

    car_surface = pygame.Surface(
        (
            66,
            38
        ),
        pygame.SRCALPHA
    )


    # shadow-ish outline
    pygame.draw.rounded_rect = getattr(
        pygame.draw,
        "rect"
    )


    # body
    pygame.draw.rect(
        car_surface,
        body_color,
        pygame.Rect(
            4,
            5,
            58,
            28
        ),
        border_radius=9
    )


    # hood
    pygame.draw.rect(
        car_surface,
        dark_color,
        pygame.Rect(
            44,
            8,
            14,
            22
        ),
        border_radius=5
    )


    # windows
    pygame.draw.rect(
        car_surface,
        WINDOW_COLOR,
        pygame.Rect(
            20,
            8,
            20,
            22
        ),
        border_radius=4
    )


    # windshield division
    pygame.draw.line(
        car_surface,
        (
            100,
            130,
            150
        ),
        (
            30,
            8
        ),
        (
            30,
            30
        ),
        2
    )


    # wheels
    for wx, wy in [
        (12, 3),
        (44, 3),
        (12, 31),
        (44, 31)
    ]:

        pygame.draw.rect(
            car_surface,
            (
                20,
                20,
                22
            ),
            pygame.Rect(
                wx,
                wy,
                10,
                4
            ),
            border_radius=2
        )


    # headlights
    pygame.draw.circle(
        car_surface,
        LIGHT_COLOR,
        (
            59,
            11
        ),
        3
    )

    pygame.draw.circle(
        car_surface,
        LIGHT_COLOR,
        (
            59,
            27
        ),
        3
    )


    # brake/rear lights
    pygame.draw.circle(
        car_surface,
        (
            230,
            55,
            55
        ),
        (
            7,
            11
        ),
        3
    )

    pygame.draw.circle(
        car_surface,
        (
            230,
            55,
            55
        ),
        (
            7,
            27
        ),
        3
    )


    if player:

        pygame.draw.rect(
            car_surface,
            (
                100,
                190,
                255
            ),
            pygame.Rect(
                4,
                5,
                58,
                28
            ),
            width=2,
            border_radius=9
        )


    return car_surface


PLAYER_CAR_SURFACE = create_car_surface(
    CAR_BODY,
    CAR_DARK,
    True
)


PARKED_CAR_SURFACE = create_car_surface(
    PARKED_BODY,
    PARKED_DARK,
    False
)


# ============================================================
# DRAW CAR AT WORLD POSITION
# ============================================================

def draw_car(
    surface,
    x,
    y,
    angle,
    car_surface,
    shadow=True
):

    # pygame rotate positive = CCW
    rotated = pygame.transform.rotate(
        car_surface,
        angle
    )


    center = world_to_screen(
        x,
        y
    )


    if shadow:

        shadow_surface = pygame.Surface(
            rotated.get_size(),
            pygame.SRCALPHA
        )


        shadow_surface.fill(
            (
                0,
                0,
                0,
                70
            )
        )


        shadow_surface.blit(
            rotated,
            (
                0,
                0
            ),
            special_flags=pygame.BLEND_RGBA_MULT
        )


        shadow_rect = (
            shadow_surface.get_rect(
                center=(
                    center[0] + 5,
                    center[1] + 6
                )
            )
        )


        surface.blit(
            shadow_surface,
            shadow_rect
        )


    rect = rotated.get_rect(
        center=center
    )


    surface.blit(
        rotated,
        rect
    )


# ============================================================
# PARKED CARS
# ============================================================

def draw_parked_cars(
    surface,
    env
):

    for index, slot in enumerate(
        env.parking_slots
    ):

        if (
            index
            ==
            env.target_slot_index
        ):
            continue


        draw_car(
            surface,
            slot.centerx,
            slot.centery,
            0,
            PARKED_CAR_SURFACE
        )


# ============================================================
# APPROACH POINT
# ============================================================

def draw_navigation_target(
    surface,
    env
):

    if env.parking_stage:

        nav_x = (
            env.parking_rect.centerx
        )

        nav_y = (
            env.parking_rect.centery
        )

        color = TARGET_BORDER

    else:

        nav_x, nav_y = (
            env.get_approach_point()
        )

        color = CYAN


    sx, sy = world_to_screen(
        nav_x,
        nav_y
    )


    pygame.draw.circle(
        surface,
        color,
        (
            sx,
            sy
        ),
        10,
        width=2
    )


    pygame.draw.line(
        surface,
        color,
        (
            sx - 13,
            sy
        ),
        (
            sx + 13,
            sy
        ),
        2
    )


    pygame.draw.line(
        surface,
        color,
        (
            sx,
            sy - 13
        ),
        (
            sx,
            sy + 13
        ),
        2
    )


# ============================================================
# TRAIL
# ============================================================

def draw_trail(
    surface,
    trail
):

    if len(trail) < 2:
        return


    points = [
        world_to_screen(
            x,
            y
        )
        for x, y in trail
    ]


    if len(points) >= 2:

        pygame.draw.lines(
            surface,
            (
                70,
                150,
                220
            ),
            False,
            points,
            2
        )


# ============================================================
# SENSORS
# ============================================================

def draw_sensors(
    surface,
    env
):

    readings = (
        env.car.get_sensor_readings(
            env.obstacles
        )
    )


    for relative_angle, reading in zip(
        env.car.sensor_angles,
        readings
    ):

        sensor_angle = (
            env.car.angle
            + relative_angle
        )


        distance = (
            reading
            *
            env.car.sensor_max_distance
        )


        rad = math.radians(
            sensor_angle
        )


        end_x = (
            env.car.x
            + math.cos(rad)
            * distance
        )


        end_y = (
            env.car.y
            - math.sin(rad)
            * distance
        )


        start_screen = world_to_screen(
            env.car.x,
            env.car.y
        )


        end_screen = world_to_screen(
            end_x,
            end_y
        )


        if reading < 0.20:

            color = RED

        elif reading < 0.50:

            color = YELLOW

        else:

            color = (
                70,
                180,
                120
            )


        pygame.draw.line(
            surface,
            color,
            start_screen,
            end_screen,
            1
        )


        pygame.draw.circle(
            surface,
            color,
            end_screen,
            3
        )


# ============================================================
# HUD
# ============================================================

def draw_hud(
    surface,
    env,
    action,
    q_values,
    paused,
    result_text,
    selected_slot,
    seed_counter,
    show_sensors,
    show_trail
):

    panel_x = 945
    panel_y = 60

    panel_w = 285
    panel_h = 600


    pygame.draw.rect(
        surface,
        PANEL,
        pygame.Rect(
            panel_x,
            panel_y,
            panel_w,
            panel_h
        ),
        border_radius=12
    )


    pygame.draw.rect(
        surface,
        (
            55,
            62,
            70
        ),
        pygame.Rect(
            panel_x,
            panel_y,
            panel_w,
            panel_h
        ),
        width=2,
        border_radius=12
    )


    x = panel_x + 20
    y = panel_y + 18


    draw_text(
        surface,
        "DQN PARKING",
        x,
        y,
        font_big,
        WHITE
    )

    y += 42


    draw_text(
        surface,
        "Autonomous Parking Agent",
        x,
        y,
        font_small,
        GRAY
    )


    y += 38


    # ========================================================
    # STATUS
    # ========================================================

    pygame.draw.rect(
        surface,
        PANEL_2,
        pygame.Rect(
            x - 5,
            y,
            250,
            150
        ),
        border_radius=8
    )


    y += 12


    side = (
        "LEFT"
        if selected_slot <= 2
        else
        "RIGHT"
        if selected_slot <= 5
        else
        "TOP"
    )


    draw_text(
        surface,
        f"Target slot: {selected_slot + 1}",
        x + 5,
        y,
        font_medium,
        TARGET_BORDER
    )

    y += 28


    draw_text(
        surface,
        f"Zona: {side}",
        x + 5,
        y,
        font_small,
        WHITE
    )

    y += 24


    draw_text(
        surface,
        f"Korak: {env.steps} / {env.max_steps}",
        x + 5,
        y,
        font_small,
        WHITE
    )

    y += 24


    draw_text(
        surface,
        f"Brzina: {env.car.speed:.2f}",
        x + 5,
        y,
        font_small,
        WHITE
    )

    y += 24


    draw_text(
        surface,
        f"Ugao: {env.car.angle:.1f} deg",
        x + 5,
        y,
        font_small,
        WHITE
    )


    y += 42


    # ========================================================
    # ACTION
    # ========================================================

    draw_text(
        surface,
        "IZABRANA AKCIJA",
        x,
        y,
        font_small,
        GRAY
    )

    y += 25


    action_name = (
        ACTION_NAMES.get(
            action,
            "-"
        )
    )


    draw_text(
        surface,
        action_name,
        x,
        y,
        font_medium,
        CYAN
    )


    y += 40


    # ========================================================
    # Q VALUES
    # ========================================================

    draw_text(
        surface,
        "Q-VREDNOSTI",
        x,
        y,
        font_small,
        GRAY
    )

    y += 23


    if q_values is not None:

        max_abs = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        q_values
                    )
                )
            )
        )


        for i in range(
            ACTION_SIZE
        ):

            q = float(
                q_values[i]
            )


            bar_width = int(
                min(
                    115,
                    abs(q)
                    /
                    max_abs
                    *
                    115
                )
            )


            color = (
                GREEN
                if i == action
                else
                (
                    90,
                    100,
                    110
                )
            )


            draw_text(
                surface,
                str(i),
                x,
                y,
                font_small,
                WHITE
            )


            pygame.draw.rect(
                surface,
                (
                    50,
                    55,
                    62
                ),
                pygame.Rect(
                    x + 22,
                    y + 4,
                    115,
                    12
                ),
                border_radius=4
            )


            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(
                    x + 22,
                    y + 4,
                    bar_width,
                    12
                ),
                border_radius=4
            )


            draw_text(
                surface,
                f"{q:6.2f}",
                x + 150,
                y,
                font_small,
                WHITE
            )


            y += 20


    # ========================================================
    # RESULT
    # ========================================================

    y += 10


    if result_text:

        if result_text == "SUCCESS":

            result_color = SUCCESS_COLOR

        elif result_text == "CRASH":

            result_color = CRASH_COLOR

        else:

            result_color = TIMEOUT_COLOR


        pygame.draw.rect(
            surface,
            result_color,
            pygame.Rect(
                x,
                y,
                230,
                40
            ),
            border_radius=8
        )


        result_surface = font_big.render(
            result_text,
            True,
            (
                15,
                20,
                20
            )
        )


        result_rect = (
            result_surface.get_rect(
                center=(
                    x + 115,
                    y + 20
                )
            )
        )


        surface.blit(
            result_surface,
            result_rect
        )


# ============================================================
# TOP HEADER
# ============================================================

def draw_header(
    surface
):

    draw_text(
        surface,
        "Autonomous Parking using Deep Q-Learning",
        30,
        14,
        font_title,
        WHITE
    )


    draw_text(
        surface,
        "Final trained policy • 31-dimensional state • 8 discrete actions",
        660,
        25,
        font_small,
        GRAY
    )


# ============================================================
# CONTROLS BAR
# ============================================================

def draw_controls(
    surface,
    show_sensors,
    show_trail,
    sim_speed
):

    text = (
        "1-8 Slot   "
        "R Restart   "
        "N Novi start   "
        "A Random slot   "
        "SPACE Pause   "
        "S Senzori   "
        "T Putanja   "
        "+/- Brzina"
    )


    pygame.draw.rect(
        surface,
        (
            20,
            23,
            28
        ),
        pygame.Rect(
            20,
            668,
            1210,
            24
        ),
        border_radius=6
    )


    draw_text(
        surface,
        text,
        35,
        670,
        font_small,
        GRAY
    )


    draw_text(
        surface,
        f"x{sim_speed}",
        1165,
        670,
        font_small,
        CYAN
    )


# ============================================================
# INIT EPISODE
# ============================================================

env = ParkingLotV2(
    start_level=0
)


selected_slot = 6

seed_counter = 0


def new_episode(
    slot=None,
    new_seed=True
):

    global selected_slot
    global seed_counter


    if slot is not None:

        selected_slot = slot


    if new_seed:

        seed_counter += 1


    seed = (
        SEED_BASE
        +
        selected_slot * 10000
        +
        seed_counter
    )


    state = reset_for_slot(
        env,
        selected_slot,
        seed
    )


    return state


state = new_episode(
    selected_slot,
    True
)


trail = []

paused = False

show_sensors = False
show_trail = True

sim_speed = 1

done = False

action = 0

q_values = np.zeros(
    ACTION_SIZE,
    dtype=np.float32
)

result_text = None


# ============================================================
# MAIN LOOP
# ============================================================

running = True


while running:

    clock.tick(
        FPS
    )


    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False


        if event.type == pygame.KEYDOWN:

            # ------------------------------------------------
            # ESC
            # ------------------------------------------------

            if event.key == pygame.K_ESCAPE:

                running = False


            # ------------------------------------------------
            # PAUSE
            # ------------------------------------------------

            elif event.key == pygame.K_SPACE:

                paused = not paused


            # ------------------------------------------------
            # RESTART SAME SEED
            # ------------------------------------------------

            elif event.key == pygame.K_r:

                seed = (
                    SEED_BASE
                    +
                    selected_slot * 10000
                    +
                    seed_counter
                )


                state = reset_for_slot(
                    env,
                    selected_slot,
                    seed
                )


                trail = []
                done = False
                result_text = None
                paused = False


            # ------------------------------------------------
            # NEW RANDOM START
            # ------------------------------------------------

            elif event.key == pygame.K_n:

                state = new_episode(
                    selected_slot,
                    True
                )

                trail = []
                done = False
                result_text = None
                paused = False


            # ------------------------------------------------
            # RANDOM SLOT
            # ------------------------------------------------

            elif event.key == pygame.K_a:

                random_slot = (
                    random.randint(
                        0,
                        7
                    )
                )


                state = new_episode(
                    random_slot,
                    True
                )

                trail = []
                done = False
                result_text = None
                paused = False


            # ------------------------------------------------
            # SENSORS
            # ------------------------------------------------

            elif event.key == pygame.K_s:

                show_sensors = (
                    not show_sensors
                )


            # ------------------------------------------------
            # TRAIL
            # ------------------------------------------------

            elif event.key == pygame.K_t:

                show_trail = (
                    not show_trail
                )


            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            elif event.key in [
                pygame.K_PLUS,
                pygame.K_EQUALS,
                pygame.K_KP_PLUS
            ]:

                sim_speed = min(
                    5,
                    sim_speed + 1
                )


            elif event.key in [
                pygame.K_MINUS,
                pygame.K_KP_MINUS
            ]:

                sim_speed = max(
                    1,
                    sim_speed - 1
                )


            # ------------------------------------------------
            # SLOT 1-8
            # ------------------------------------------------

            elif pygame.K_1 <= event.key <= pygame.K_8:

                slot = (
                    event.key
                    - pygame.K_1
                )


                state = new_episode(
                    slot,
                    True
                )

                trail = []
                done = False
                result_text = None
                paused = False


    # ========================================================
    # SIMULATION
    # ========================================================

    if (
        not paused
        and
        not done
    ):

        for _ in range(
            sim_speed
        ):

            safe_actions = (
                get_safe_actions(
                    env
                )
            )


            (
                action,
                q_values
            ) = choose_action(
                state,
                safe_actions
            )


            (
                next_state,
                reward,
                done
            ) = env.step(
                action
            )


            state = (
                next_state
            )


            trail.append(
                (
                    env.car.x,
                    env.car.y
                )
            )


            if len(trail) > 500:

                trail.pop(0)


            if done:

                if env.car.parked:

                    result_text = (
                        "SUCCESS"
                    )

                elif env.car.crashed:

                    result_text = (
                        "CRASH"
                    )

                else:

                    result_text = (
                        "TIMEOUT"
                    )


                break


    # ========================================================
    # DRAW
    # ========================================================

    screen.fill(
        BACKGROUND
    )


    draw_header(
        screen
    )


    draw_parking_background(
        screen
    )


    draw_slots(
        screen,
        env
    )


    if show_trail:

        draw_trail(
            screen,
            trail
        )


    draw_navigation_target(
        screen,
        env
    )


    draw_parked_cars(
        screen,
        env
    )


    if show_sensors:

        draw_sensors(
            screen,
            env
        )


    draw_car(
        screen,
        env.car.x,
        env.car.y,
        env.car.angle,
        PLAYER_CAR_SURFACE
    )


    draw_hud(
        screen,
        env,
        action,
        q_values,
        paused,
        result_text,
        selected_slot,
        seed_counter,
        show_sensors,
        show_trail
    )


    draw_controls(
        screen,
        show_sensors,
        show_trail,
        sim_speed
    )


    # ========================================================
    # PAUSED OVERLAY
    # ========================================================

    if paused:

        pause_surface = pygame.Surface(
            (
                WIDTH,
                HEIGHT
            ),
            pygame.SRCALPHA
        )


        pause_surface.fill(
            (
                0,
                0,
                0,
                80
            )
        )


        screen.blit(
            pause_surface,
            (
                PARKING_X,
                PARKING_Y
            )
        )


        pause_text = font_title.render(
            "PAUSED",
            True,
            WHITE
        )


        pause_rect = (
            pause_text.get_rect(
                center=(
                    PARKING_X
                    + WIDTH // 2,

                    PARKING_Y
                    + HEIGHT // 2
                )
            )
        )


        screen.blit(
            pause_text,
            pause_rect
        )


    pygame.display.flip()


pygame.quit()