import math
import random
import pygame

from parking_env_final import (
    Car,
    WIDTH,
    HEIGHT
)


class ParkingLotV2:

    def __init__(self, start_level=0):

        # ====================================================
        # START LEVEL
        #
        # 0 = potpuno isti start
        # 1 = mala randomizacija
        # 2 = veca randomizacija
        # ====================================================

        self.start_level = start_level

        self.car = None

        self.steps = 0
        self.max_steps = 700

        self.target_slot_index = 0
        self.parking_rect = None

        self.obstacles = []

        self.parking_stage = False

        self.stuck_counter = 0

        self.previous_nav_distance = 0.0
        self.previous_balance_error = 0.0
        self.previous_outside_error = 0.0
        self.previous_corners_inside = 0

        # ====================================================
        # NOVI PARKING
        #
        # 3 mesta LEVO
        # 3 mesta DESNO
        # 2 mesta GORE
        #
        # Ukupno 8.
        # Jedno je random target, ostala su zauzeta.
        # ====================================================

        self.parking_slots = [

            # LEVO
            pygame.Rect(
                50,
                100,
                130,
                70
            ),

            pygame.Rect(
                50,
                240,
                130,
                70
            ),

            pygame.Rect(
                50,
                380,
                130,
                70
            ),

            # DESNO
            pygame.Rect(
                720,
                100,
                130,
                70
            ),

            pygame.Rect(
                720,
                240,
                130,
                70
            ),

            pygame.Rect(
                720,
                380,
                130,
                70
            ),

            # GORE
            pygame.Rect(
                290,
                40,
                130,
                70
            ),

            pygame.Rect(
                480,
                40,
                130,
                70
            ),
        ]

        self.reset()


    # ========================================================
    # PARKED CAR
    # ========================================================

    def create_parked_car(self, slot):

        return pygame.Rect(
            slot.centerx - 29,
            slot.centery - 17,
            58,
            34
        )


    # ========================================================
    # START POSITION
    # ========================================================

    def get_start_position(self):

        # ----------------------------------------------------
        # LEVEL 0
        # potpuno isti start
        # ----------------------------------------------------

        if self.start_level == 0:

            x = 450.0
            y = 525.0
            angle = 90.0


        # ----------------------------------------------------
        # LEVEL 1
        # mala randomizacija
        # ----------------------------------------------------

        elif self.start_level == 1:

            x = (
                450.0
                + random.uniform(
                    -15,
                    15
                )
            )

            y = (
                525.0
                + random.uniform(
                    -10,
                    10
                )
            )

            angle = (
                90.0
                + random.uniform(
                    -5,
                    5
                )
            )


        # ----------------------------------------------------
        # LEVEL 2
        # veca randomizacija
        # ----------------------------------------------------

        else:

            x = (
                450.0
                + random.uniform(
                    -30,
                    30
                )
            )

            y = (
                525.0
                + random.uniform(
                    -20,
                    20
                )
            )

            angle = (
                90.0
                + random.uniform(
                    -10,
                    10
                )
            )


        return (
            x,
            y,
            angle
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.steps = 0

        self.parking_stage = False
        self.stuck_counter = 0


        # ====================================================
        # RANDOM SLOBODNO MESTO
        # ====================================================

        self.target_slot_index = (
            random.randrange(
                len(
                    self.parking_slots
                )
            )
        )

        self.parking_rect = (
            self.parking_slots[
                self.target_slot_index
            ]
        )


        # ====================================================
        # SVA OSTALA MESTA SU ZAUZETA
        # ====================================================

        self.obstacles = []

        for index, slot in enumerate(
            self.parking_slots
        ):

            if (
                index
                ==
                self.target_slot_index
            ):
                continue

            self.obstacles.append(
                self.create_parked_car(
                    slot
                )
            )


        # ====================================================
        # AUTO
        # ====================================================

        (
            start_x,
            start_y,
            start_angle
        ) = (
            self.get_start_position()
        )


        self.car = Car(
            start_x,
            start_y
        )

        self.car.angle = (
            start_angle % 360
        )

        self.car.speed = 0.0

        self.car.crashed = False
        self.car.parked = False


        # ====================================================
        # HISTORY
        # ====================================================

        self.previous_nav_distance = (
            self.distance_to_navigation_target()
        )


        geometry = (
            self.get_slot_geometry()
        )


        self.previous_balance_error = (
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


        self.previous_outside_error = (
            self.get_outside_error(
                geometry
            )
        )


        self.previous_corners_inside = (
            self.corners_inside_parking()
        )


        return self.get_state()


    # ========================================================
    # SLOT SIDE
    #
    # 0,1,2 = LEFT
    # 3,4,5 = RIGHT
    # 6,7 = TOP
    # ========================================================

    def get_target_side(self):

        if self.target_slot_index <= 2:
            return "LEFT"

        if self.target_slot_index <= 5:
            return "RIGHT"

        return "TOP"


    # ========================================================
    # APPROACH POINT
    # ========================================================

    def get_approach_point(self):

        side = (
            self.get_target_side()
        )


        # ----------------------------------------------------
        # LEVO
        #
        # Dodjemo prvo desno od mesta.
        # ----------------------------------------------------

        if side == "LEFT":

            return (
                245.0,
                float(
                    self.parking_rect.centery
                )
            )


        # ----------------------------------------------------
        # DESNO
        #
        # Dodjemo prvo levo od mesta.
        # ----------------------------------------------------

        if side == "RIGHT":

            return (
                655.0,
                float(
                    self.parking_rect.centery
                )
            )


        # ----------------------------------------------------
        # GORE
        #
        # Dodjemo ispod mesta.
        # ----------------------------------------------------

        return (
            float(
                self.parking_rect.centerx
            ),
            160.0
        )


    # ========================================================
    # NAVIGATION TARGET
    # ========================================================

    def get_navigation_target(self):

        if self.parking_stage:

            return (
                float(
                    self.parking_rect.centerx
                ),
                float(
                    self.parking_rect.centery
                )
            )


        return self.get_approach_point()


    # ========================================================
    # NAV DISTANCE
    # ========================================================

    def distance_to_navigation_target(self):

        target_x, target_y = (
            self.get_navigation_target()
        )


        dx = (
            target_x
            - self.car.x
        )

        dy = (
            target_y
            - self.car.y
        )


        return math.sqrt(
            dx * dx
            +
            dy * dy
        )


    # ========================================================
    # TARGET CENTER DISTANCE
    # ========================================================

    def distance_to_parking(self):

        dx = (
            self.parking_rect.centerx
            - self.car.x
        )

        dy = (
            self.parking_rect.centery
            - self.car.y
        )


        return math.sqrt(
            dx * dx
            +
            dy * dy
        )


    # ========================================================
    # HORIZONTAL ANGLE ERROR
    #
    # Parking mesta su horizontalna.
    #
    # 0 stepeni i 180 stepeni prihvatamo kao pravilno
    # parkiranje.
    # ========================================================

    def horizontal_angle_error(self):

        angle = (
            self.car.angle % 360
        )


        error_0 = (
            (
                angle
                + 180
            )
            % 360
        ) - 180


        error_180 = (
            (
                angle
                - 180
                + 180
            )
            % 360
        ) - 180


        if (
            abs(error_0)
            <=
            abs(error_180)
        ):

            return error_0


        return error_180


    # ========================================================
    # COLLISION
    # ========================================================

    def check_collision(self):

        corners = (
            self.car.get_corners()
        )


        # ----------------------------------------------------
        # WALL
        # ----------------------------------------------------

        for x, y in corners:

            if (
                x < 22
                or x > WIDTH - 22
                or y < 22
                or y > HEIGHT - 22
            ):

                return True


        # ----------------------------------------------------
        # PARKED CARS
        # ----------------------------------------------------

        for obstacle in self.obstacles:

            safe_obstacle = (
                obstacle.inflate(
                    8,
                    8
                )
            )


            for x, y in corners:

                if safe_obstacle.collidepoint(
                    x,
                    y
                ):

                    return True


            if safe_obstacle.collidepoint(
                self.car.x,
                self.car.y
            ):

                return True


        return False


    # ========================================================
    # CORNERS INSIDE TARGET
    # ========================================================

    def corners_inside_parking(self):

        count = 0


        for x, y in self.car.get_corners():

            if self.parking_rect.collidepoint(
                x,
                y
            ):

                count += 1


        return count


    # ========================================================
    # SUCCESS
    # ========================================================

    def check_parking(self):

        if (
            self.corners_inside_parking()
            < 4
        ):

            return False


        if abs(
            self.car.speed
        ) > 0.30:

            return False


        if abs(
            self.horizontal_angle_error()
        ) > 15.0:

            return False


        return True


    # ========================================================
    # SEMANTIC SENSOR TYPES
    #
    # 0.0 = nothing
    # 0.5 = wall
    # 1.0 = parked car
    # ========================================================

    def get_semantic_sensor_types(self):

        result = []


        for relative_angle in (
            self.car.sensor_angles
        ):

            sensor_angle = (
                self.car.angle
                + relative_angle
            ) % 360


            rad = math.radians(
                sensor_angle
            )


            object_type = 0.0


            for distance in range(
                0,
                self.car.sensor_max_distance + 1,
                4
            ):

                sensor_x = (
                    self.car.x
                    + math.cos(rad)
                    * distance
                )

                sensor_y = (
                    self.car.y
                    - math.sin(rad)
                    * distance
                )


                # --------------------------------------------
                # WALL
                # --------------------------------------------

                if (
                    sensor_x < 20
                    or sensor_x > WIDTH - 20
                    or sensor_y < 20
                    or sensor_y > HEIGHT - 20
                ):

                    object_type = 0.5
                    break


                # --------------------------------------------
                # PARKED CAR
                # --------------------------------------------

                hit_car = False


                for obstacle in self.obstacles:

                    if obstacle.collidepoint(
                        sensor_x,
                        sensor_y
                    ):

                        object_type = 1.0
                        hit_car = True
                        break


                if hit_car:
                    break


            result.append(
                object_type
            )


        return result


    # ========================================================
    # SEMANTIC PROXIMITY
    #
    # + = parked car
    # - = wall
    # 0 = nothing
    # ========================================================

    def get_semantic_sensor_features(self):

        distances = (
            self.car.get_sensor_readings(
                self.obstacles
            )
        )

        types = (
            self.get_semantic_sensor_types()
        )


        features = []


        for distance, object_type in zip(
            distances,
            types
        ):

            proximity = (
                1.0
                - distance
            )


            if object_type == 1.0:

                features.append(
                    proximity
                )


            elif object_type == 0.5:

                features.append(
                    -proximity
                )


            else:

                features.append(
                    0.0
                )


        return features


    # ========================================================
    # SLOT GEOMETRY
    # ========================================================

    def get_slot_geometry(self):

        corners = (
            self.car.get_corners()
        )


        xs = [
            point[0]
            for point in corners
        ]

        ys = [
            point[1]
            for point in corners
        ]


        car_left = min(xs)
        car_right = max(xs)

        car_top = min(ys)
        car_bottom = max(ys)


        slot = (
            self.parking_rect
        )


        # ====================================================
        # NORMALIZED GAPS
        # ====================================================

        left_gap = (
            car_left
            - slot.left
        ) / slot.width


        right_gap = (
            slot.right
            - car_right
        ) / slot.width


        top_gap = (
            car_top
            - slot.top
        ) / slot.height


        bottom_gap = (
            slot.bottom
            - car_bottom
        ) / slot.height


        left_gap = max(
            -1.0,
            min(
                1.0,
                left_gap
            )
        )

        right_gap = max(
            -1.0,
            min(
                1.0,
                right_gap
            )
        )

        top_gap = max(
            -1.0,
            min(
                1.0,
                top_gap
            )
        )

        bottom_gap = max(
            -1.0,
            min(
                1.0,
                bottom_gap
            )
        )


        horizontal_balance = (
            left_gap
            - right_gap
        )


        vertical_balance = (
            top_gap
            - bottom_gap
        )


        return {

            "left_gap":
                left_gap,

            "right_gap":
                right_gap,

            "top_gap":
                top_gap,

            "bottom_gap":
                bottom_gap,

            "horizontal_balance":
                horizontal_balance,

            "vertical_balance":
                vertical_balance
        }


    # ========================================================
    # OUTSIDE ERROR
    # ========================================================

    def get_outside_error(
        self,
        geometry
    ):

        return (

            max(
                0.0,
                -geometry[
                    "left_gap"
                ]
            )

            +

            max(
                0.0,
                -geometry[
                    "right_gap"
                ]
            )

            +

            max(
                0.0,
                -geometry[
                    "top_gap"
                ]
            )

            +

            max(
                0.0,
                -geometry[
                    "bottom_gap"
                ]
            )
        )


    # ========================================================
    # STATE = 31
    #
    # 9 navigation
    # 8 distance
    # 8 semantic
    # 6 slot geometry
    # ========================================================

    def get_state(self):

        nav_x, nav_y = (
            self.get_navigation_target()
        )


        dx = (
            nav_x
            - self.car.x
        )

        dy = (
            nav_y
            - self.car.y
        )


        nav_distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        rad = math.radians(
            self.car.angle
        )


        angle_error = (
            self.horizontal_angle_error()
        )


        state = [

            self.car.x / WIDTH,

            self.car.y / HEIGHT,

            math.cos(rad),

            math.sin(rad),

            self.car.speed
            / self.car.max_speed,

            dx / WIDTH,

            dy / HEIGHT,

            nav_distance
            /
            math.sqrt(
                WIDTH * WIDTH
                +
                HEIGHT * HEIGHT
            ),

            angle_error / 180.0
        ]


        # ====================================================
        # 8 DISTANCE SENSORS
        # ====================================================

        state.extend(
            self.car.get_sensor_readings(
                self.obstacles
            )
        )


        # ====================================================
        # 8 SEMANTIC
        # ====================================================

        state.extend(
            self.get_semantic_sensor_features()
        )


        # ====================================================
        # 6 SLOT GEOMETRY
        # ====================================================

        geometry = (
            self.get_slot_geometry()
        )


        state.extend([

            geometry[
                "left_gap"
            ],

            geometry[
                "right_gap"
            ],

            geometry[
                "top_gap"
            ],

            geometry[
                "bottom_gap"
            ],

            geometry[
                "horizontal_balance"
            ],

            geometry[
                "vertical_balance"
            ]
        ])


        return state


    # ========================================================
    # APPROACH REACHED
    # ========================================================

    def check_approach_reached(self):

        approach_x, approach_y = (
            self.get_approach_point()
        )


        dx = (
            approach_x
            - self.car.x
        )

        dy = (
            approach_y
            - self.car.y
        )


        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )


        return (
            distance < 55.0
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        self.steps += 1


        previous_x = (
            self.car.x
        )

        previous_y = (
            self.car.y
        )


        old_nav_distance = (
            self.distance_to_navigation_target()
        )


        old_geometry = (
            self.get_slot_geometry()
        )


        old_balance_error = (

            abs(
                old_geometry[
                    "horizontal_balance"
                ]
            )

            +

            abs(
                old_geometry[
                    "vertical_balance"
                ]
            )
        )


        old_outside_error = (
            self.get_outside_error(
                old_geometry
            )
        )


        old_corners = (
            self.corners_inside_parking()
        )


        # ====================================================
        # ACTION
        # ====================================================

        self.car.apply_action(
            action
        )


        # ====================================================
        # TERMINAL CHECK
        # ====================================================

        collision = (
            self.check_collision()
        )


        parked = False


        if not collision:

            parked = (
                self.check_parking()
            )


        # ====================================================
        # BASE REWARD
        # ====================================================

        reward = -0.02
        done = False


        # ====================================================
        # CRASH
        # ====================================================

        if collision:

            self.car.crashed = True

            reward = -200.0

            done = True

            return (
                self.get_state(),
                reward,
                done
            )


        # ====================================================
        # SUCCESS
        # ====================================================

        if parked:

            self.car.parked = True

            reward = 300.0

            done = True

            return (
                self.get_state(),
                reward,
                done
            )


        # ====================================================
        # APPROACH -> PARKING STAGE
        # ====================================================

        if (
            not self.parking_stage
            and
            self.check_approach_reached()
        ):

            self.parking_stage = True

            reward += 12.0


            # nav target se upravo promenio,
            # pa ne koristimo stari/new distance
            # preko promene stage-a.

            old_nav_distance = (
                self.distance_to_navigation_target()
            )


        # ====================================================
        # NAVIGATION PROGRESS
        # ====================================================

        new_nav_distance = (
            self.distance_to_navigation_target()
        )


        progress = (
            old_nav_distance
            - new_nav_distance
        )


        reward += (
            progress
            * 0.12
        )


        # ====================================================
        # SAFETY AROUND PARKED CARS
        # ====================================================

        distances = (
            self.car.get_sensor_readings(
                self.obstacles
            )
        )


        types = (
            self.get_semantic_sensor_types()
        )


        speed = abs(
            self.car.speed
        )


        for distance, object_type in zip(
            distances,
            types
        ):

            if object_type != 1.0:
                continue


            proximity = (
                1.0
                - distance
            )


            # Tek kad je stvarno blizu.
            if proximity > 0.72:

                reward -= (

                    (
                        proximity
                        - 0.72
                    )

                    * 2.0

                    * (
                        1.0
                        + speed * 0.5
                    )
                )


        # ====================================================
        # PARKING GEOMETRY
        #
        # Samo blizu targeta.
        #
        # Bitno:
        # NEMA ogromne konstantne nagrade svaki frame.
        # Nagradjujemo POBOLJSANJE.
        # ====================================================

        center_distance = (
            self.distance_to_parking()
        )


        new_geometry = (
            self.get_slot_geometry()
        )


        new_balance_error = (

            abs(
                new_geometry[
                    "horizontal_balance"
                ]
            )

            +

            abs(
                new_geometry[
                    "vertical_balance"
                ]
            )
        )


        new_outside_error = (
            self.get_outside_error(
                new_geometry
            )
        )


        new_corners = (
            self.corners_inside_parking()
        )


        if (
            center_distance < 120
            or new_corners > 0
        ):

            balance_improvement = (
                old_balance_error
                - new_balance_error
            )


            outside_improvement = (
                old_outside_error
                - new_outside_error
            )


            reward += (
                balance_improvement
                * 1.0
            )


            reward += (
                outside_improvement
                * 0.8
            )


            # -----------------------------------------------
            # Promena broja uglova unutra.
            #
            # Nagrada samo kad ZAISTA udje jos jedan ugao.
            # Nema +6 svaki frame dok stoji.
            # -----------------------------------------------

            corner_change = (
                new_corners
                - old_corners
            )


            reward += (
                corner_change
                * 2.0
            )


        # ====================================================
        # SLOW DOWN NEAR TARGET
        # ====================================================

        if center_distance < 110:

            if speed > 1.2:

                reward -= (
                    speed - 1.2
                ) * 0.20


        # ====================================================
        # AKO JE POTPUNO UNUTRA,
        # ali jos nije dovoljno spor
        # ====================================================

        if new_corners == 4:

            if speed <= 0.60:

                reward += 0.10

            if speed > 1.0:

                reward -= 0.30


        # ====================================================
        # STUCK
        # ====================================================

        moved = math.sqrt(

            (
                self.car.x
                - previous_x
            ) ** 2

            +

            (
                self.car.y
                - previous_y
            ) ** 2
        )


        if (
            moved < 0.08
            and
            speed < 0.12
        ):

            self.stuck_counter += 1

        else:

            self.stuck_counter = max(
                0,
                self.stuck_counter - 2
            )


        if self.stuck_counter > 25:

            reward -= 0.08


        if self.stuck_counter > 50:

            reward -= 0.20


        # ====================================================
        # TIMEOUT
        # ====================================================

        if self.steps >= self.max_steps:

            reward -= 80.0

            done = True


        # ====================================================
        # HISTORY
        # ====================================================

        self.previous_nav_distance = (
            new_nav_distance
        )

        self.previous_balance_error = (
            new_balance_error
        )

        self.previous_outside_error = (
            new_outside_error
        )

        self.previous_corners_inside = (
            new_corners
        )


        return (
            self.get_state(),
            reward,
            done
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    env = ParkingLotV2(
        start_level=0
    )


    print("=" * 70)
    print("PARKING LOT V2 TEST")
    print("=" * 70)


    for i in range(10):

        state = env.reset()

        print(
            f"{i + 1:02d} | "
            f"state={len(state)} | "
            f"target={env.target_slot_index} | "
            f"side={env.get_target_side()} | "
            f"start=({env.car.x:.1f}, {env.car.y:.1f}) | "
            f"angle={env.car.angle:.1f} | "
            f"approach={env.get_approach_point()}"
        )