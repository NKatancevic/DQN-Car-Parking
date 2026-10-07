import math
import random
import pygame


WIDTH = 900
HEIGHT = 600




class Car:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.angle = 0
        self.speed = 0

        self.width = 50
        self.height = 25

        self.max_speed = 4
        self.acceleration = 0.15
        self.friction = 0.05
        self.rotation_speed = 3

        self.crashed = False
        self.parked = False


        self.sensor_angles = [
            -90,  
            -45,  
            0, 
            45,  
            90,  

            135, 
            180,  
            -135 
        ]

        self.sensor_max_distance = 180




    def apply_action(self, action):

        if self.crashed or self.parked:
            return



        if action in [1, 3, 4]:

            self.speed += self.acceleration

        elif action in [2, 5, 6]:

            self.speed -= self.acceleration

        elif action == 7:

            brake_strength = 0.25

            if self.speed > 0:

                self.speed -= brake_strength

                if self.speed < 0:
                    self.speed = 0

            elif self.speed < 0:

                self.speed += brake_strength

                if self.speed > 0:
                    self.speed = 0

        else:

            if self.speed > 0:

                self.speed -= self.friction

                if self.speed < 0:
                    self.speed = 0

            elif self.speed < 0:

                self.speed += self.friction

                if self.speed > 0:
                    self.speed = 0


        self.speed = min(
            self.speed,
            self.max_speed
        )

        self.speed = max(
            self.speed,
            -self.max_speed / 2
        )




        if (
            action in [3, 5]
            and abs(self.speed) > 0.1
        ):

            if self.speed > 0:
                self.angle += self.rotation_speed
            else:
                self.angle -= self.rotation_speed


        if (
            action in [4, 6]
            and abs(self.speed) > 0.1
        ):

            if self.speed > 0:
                self.angle -= self.rotation_speed
            else:
                self.angle += self.rotation_speed


        self.angle %= 360




        rad = math.radians(
            self.angle
        )

        self.x += (
            math.cos(rad)
            * self.speed
        )

        self.y -= (
            math.sin(rad)
            * self.speed
        )




    def get_corners(self):

        rad = math.radians(
            self.angle
        )

        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        half_w = self.width / 2
        half_h = self.height / 2


        local_corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]


        corners = []


        for local_x, local_y in local_corners:

            rotated_x = (
                local_x * cos_a
                - local_y * sin_a
            )

            rotated_y = (
                local_x * sin_a
                + local_y * cos_a
            )


            world_x = (
                self.x
                + rotated_x
            )

            world_y = (
                self.y
                - rotated_y
            )


            corners.append(
                (
                    world_x,
                    world_y
                )
            )


        return corners



    def get_sensor_readings(
        self,
        obstacles
    ):

        readings = []


        for relative_angle in self.sensor_angles:

            sensor_angle = (
                self.angle
                + relative_angle
            ) % 360


            rad = math.radians(
                sensor_angle
            )


            distance = (
                self.sensor_max_distance
            )


            for d in range(
                0,
                self.sensor_max_distance + 1,
                4
            ):

                sensor_x = (
                    self.x
                    + math.cos(rad) * d
                )

                sensor_y = (
                    self.y
                    - math.sin(rad) * d
                )


                hit = False



                if (
                    sensor_x < 20
                    or sensor_x > WIDTH - 20
                    or sensor_y < 20
                    or sensor_y > HEIGHT - 20
                ):

                    distance = d
                    hit = True


                if not hit:

                    for obstacle in obstacles:

                        if obstacle.collidepoint(
                            sensor_x,
                            sensor_y
                        ):

                            distance = d
                            hit = True
                            break


                if hit:
                    break


            readings.append(
                distance
                / self.sensor_max_distance
            )


        return readings





class RealParkingEnv:

    def __init__(
        self,
        difficulty="easy"
    ):

        self.difficulty = difficulty

        self.car = None

        self.steps = 0
        self.max_steps = 600




        self.parking_slots = [

            pygame.Rect(
                690,
                65,
                130,
                70
            ),

            pygame.Rect(
                690,
                165,
                130,
                70
            ),

            pygame.Rect(
                690,
                265,
                130,
                70
            ),

            pygame.Rect(
                690,
                365,
                130,
                70
            ),

            pygame.Rect(
                690,
                465,
                130,
                70
            )
        ]


        self.target_slot_index = 2

        self.parking_rect = (
            self.parking_slots[
                self.target_slot_index
            ]
        )


        self.obstacles = []


        self.previous_distance = None

        self.previous_angle_difference = None

        self.previous_corners_inside = 0

        self.previous_lateral_error = None


        self.reset()



    def set_difficulty(
        self,
        difficulty
    ):

        self.difficulty = difficulty




    def create_parked_car(
        self,
        slot
    ):

        # Malo manji od parking mesta
        return pygame.Rect(
            slot.centerx - 29,
            slot.centery - 17,
            58,
            34
        )



    def reset(self):

        self.steps = 0




        if self.difficulty == "easy":

            self.target_slot_index = (
                random.choice(
                    [1, 2, 3]
                )
            )

            target_slot = (
                self.parking_slots[
                    self.target_slot_index
                ]
            )


            start_x = random.randint(
                500,
                570
            )

            start_y = (
                target_slot.centery
                + random.randint(
                    -25,
                    25
                )
            )

            start_angle = random.randint(
                -15,
                15
            )

            self.max_steps = 450




        elif self.difficulty == "medium":

            self.target_slot_index = (
                random.choice(
                    [1, 2, 3]
                )
            )

            target_slot = (
                self.parking_slots[
                    self.target_slot_index
                ]
            )


            start_x = random.randint(
                430,
                560
            )

            start_y = (
                target_slot.centery
                + random.randint(
                    -65,
                    65
                )
            )

            start_y = max(
                70,
                min(
                    HEIGHT - 70,
                    start_y
                )
            )


            start_angle = random.randint(
                -25,
                25
            )

            self.max_steps = 550


  

        else:

            self.target_slot_index = (
                random.randint(
                    0,
                    len(
                        self.parking_slots
                    ) - 1
                )
            )

            target_slot = (
                self.parking_slots[
                    self.target_slot_index
                ]
            )


            start_x = random.randint(
                350,
                550
            )


            start_y = random.randint(
                90,
                HEIGHT - 90
            )


            start_angle = random.randint(
                -35,
                35
            )

            self.max_steps = 650




        self.parking_rect = (
            self.parking_slots[
                self.target_slot_index
            ]
        )




        self.obstacles = []


        for index, slot in enumerate(
            self.parking_slots
        ):

            if (
                index
                == self.target_slot_index
            ):

                continue


            parked_car = (
                self.create_parked_car(
                    slot
                )
            )

            self.obstacles.append(
                parked_car
            )




        self.car = Car(
            start_x,
            start_y
        )

        self.car.angle = (
            start_angle % 360
        )

        self.car.speed = 0

        self.car.crashed = False
        self.car.parked = False




        self.previous_distance = (
            self.distance_to_parking()
        )


        self.previous_angle_difference = (
            self.angle_difference()
        )


        self.previous_corners_inside = (
            self.corners_inside_parking()
        )


        self.previous_lateral_error = abs(
            self.car.y
            - self.parking_rect.centery
        )


        return self.get_state()




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
            + dy * dy
        )




    def angle_difference(self):

        return abs(
            (
                (
                    self.car.angle
                    + 180
                )
                % 360
            )
            - 180
        )




    def corners_inside_parking(self):

        inside = 0


        for x, y in (
            self.car.get_corners()
        ):

            if (
                self.parking_rect
                .collidepoint(
                    x,
                    y
                )
            ):

                inside += 1


        return inside



    def check_collision(self):

        corners = (
            self.car.get_corners()
        )



        for x, y in corners:

            if (
                x < 20
                or x > WIDTH - 20
                or y < 20
                or y > HEIGHT - 20
            ):

                return True



        for obstacle in self.obstacles:

            for x, y in corners:

                if obstacle.collidepoint(
                    x,
                    y
                ):

                    return True



            if obstacle.collidepoint(
                self.car.x,
                self.car.y
            ):

                return True


        return False




    def check_parking(self):

        if (
            self.corners_inside_parking()
            < 4
        ):

            return False


        if abs(
            self.car.speed
        ) > 0.3:

            return False


        if (
            self.angle_difference()
            > 15
        ):

            return False


        self.car.parked = True

        return True




    def get_state(self):

        target_x = (
            self.parking_rect.centerx
        )

        target_y = (
            self.parking_rect.centery
        )


        dx = (
            target_x
            - self.car.x
        )

        dy = (
            target_y
            - self.car.y
        )


        distance = math.sqrt(
            dx * dx
            + dy * dy
        )


        rad = math.radians(
            self.car.angle
        )


        signed_angle_difference = (
            (
                self.car.angle
                + 180
            )
            % 360
        ) - 180


        state = [

            self.car.x / WIDTH,

            self.car.y / HEIGHT,

            math.cos(rad),

            math.sin(rad),

            self.car.speed
            / self.car.max_speed,

            dx / WIDTH,

            dy / HEIGHT,

            distance
            / math.sqrt(
                WIDTH * WIDTH
                + HEIGHT * HEIGHT
            ),

            signed_angle_difference
            / 180
        ]


        sensors = (
            self.car
            .get_sensor_readings(
                self.obstacles
            )
        )


        state.extend(
            sensors
        )


        return state




    def step(self, action):

        self.steps += 1

        previous_speed = abs(self.car.speed)

        self.car.apply_action(
            action
        )

        collision = (
            self.check_collision()
        )

        parked = False

        if not collision:

            parked = (
                self.check_parking()
            )


        reward = -0.02
        done = False



        current_distance = (
            self.distance_to_parking()
        )

        distance_improvement = (
            self.previous_distance
            - current_distance
        )

        reward += (
            distance_improvement
            * 0.03
        )

        self.previous_distance = (
            current_distance
        )



        current_angle_difference = (
            self.angle_difference()
        )

        angle_improvement = (
            self.previous_angle_difference
            - current_angle_difference
        )

        reward += (
            angle_improvement
            * 0.015
        )

        self.previous_angle_difference = (
            current_angle_difference
        )



        current_lateral_error = abs(
            self.car.y
            - self.parking_rect.centery
        )

        lateral_improvement = (
            self.previous_lateral_error
            - current_lateral_error
        )

        reward += (
            lateral_improvement
            * 0.025
        )

        self.previous_lateral_error = (
            current_lateral_error
        )



        current_corners = (
            self.corners_inside_parking()
        )

        corner_progress = (
            current_corners
            - self.previous_corners_inside
        )

        reward += (
            corner_progress
            * 2.0
        )

        self.previous_corners_inside = (
            current_corners
        )



        if current_distance < 120:

            current_speed = abs(
                self.car.speed
            )

            if current_speed > 1.0:

                reward -= (
                    current_speed - 1.0
                ) * 0.08



        if (
            abs(self.car.speed) < 0.1
            and
            self.parking_rect.collidepoint(
                self.car.x,
                self.car.y
            )
            and
            not parked
        ):

            reward -= 0.10



        if collision:

            self.car.crashed = True

            reward = -80

            done = True

        elif parked:

            reward = 150

            done = True

        elif (
            self.steps
            >= self.max_steps
        ):

            reward = -25

            done = True

        return (
            self.get_state(),
            reward,
            done
        )
