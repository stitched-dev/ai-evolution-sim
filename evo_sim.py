import pygame
import random
import math
import numpy as np
import sys

# === Constants ===
WIDTH, HEIGHT = 800, 600
NUM_CREATURES = 30
NUM_PREDATORS = 2
NUM_FOOD = 60
GENERATION_TIME = 30  # seconds
FPS = 60
MIN_FPS, MAX_FPS = 10, 300

CREATURE_INPUTS = 3
PREDATOR_INPUTS = 4
HIDDEN_SIZE = 6
OUTPUT_SIZE = 2

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

def angle_between(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# === Neural Network ===
class NeuralNetwork:
    def __init__(self, input_size):
        self.w1 = np.random.randn(input_size, HIDDEN_SIZE)
        self.w2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1))
        o = np.tanh(np.dot(h, self.w2))
        return o

    def clone(self):
        clone = NeuralNetwork(self.w1.shape[0])
        clone.w1 = np.copy(self.w1)
        clone.w2 = np.copy(self.w2)
        return clone

    def mutate(self, rate=0.1):
        self.w1 += np.random.randn(*self.w1.shape) * rate
        self.w2 += np.random.randn(*self.w2.shape) * rate

# === Food ===
class Food:
    def __init__(self):
        self.pos = [random.randint(0, WIDTH), random.randint(0, HEIGHT)]

    def draw(self):
        pygame.draw.circle(screen, (0, 255, 0), self.pos, 4)

# === Creature (Prey) ===
class Creature:
    def __init__(self, brain=None):
        self.pos = [random.randint(0, WIDTH), random.randint(0, HEIGHT)]
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = 100
        self.brain = brain.clone() if brain else NeuralNetwork(CREATURE_INPUTS)
        self.score = 0
        self.alive = True

    def update(self, foods):
        if not self.alive:
            return
        if self.energy <= 0:
            self.alive = False
            return
        if not foods:
            return
        closest = min(foods, key=lambda f: distance(self.pos, f.pos))
        angle_to = angle_between(self.pos, closest.pos)
        dist = distance(self.pos, closest.pos) / math.hypot(WIDTH, HEIGHT)
        angle_diff = math.sin(angle_to - self.angle)
        inputs = np.array([angle_diff, dist, self.energy / 100])
        output = self.brain.forward(inputs)

        self.angle += output[0] * 0.2
        speed = max(0, output[1]) * 2
        self.pos[0] += math.cos(self.angle) * speed
        self.pos[1] += math.sin(self.angle) * speed
        self.pos[0] %= WIDTH
        self.pos[1] %= HEIGHT
        self.energy -= 0.1

        if distance(self.pos, closest.pos) < 10:
            self.energy += 20
            self.score += 1
            foods.remove(closest)

    def draw(self):
        if self.alive:
            pygame.draw.circle(screen, (0, 0, 255), (int(self.pos[0]), int(self.pos[1])), 6)

# === Predator ===
class Predator:
    def __init__(self, brain=None):
        self.pos = [random.randint(0, WIDTH), random.randint(0, HEIGHT)]
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = 100
        self.kills = 0
        self.brain = brain.clone() if brain else NeuralNetwork(PREDATOR_INPUTS)
        self.alive = True
        self.feeding = False
        self.feed_timer = 0
        self.feed_duration = 60  # frames
        self.last_kill_pos = None

    def update(self, creatures, others):
        if not self.alive:
            return
        if self.energy <= 0:
            self.alive = False
            return

        if self.feeding:
            if distance(self.pos, self.last_kill_pos) < 10:
                self.feed_timer += 1
                if self.feed_timer >= self.feed_duration:
                    self.energy += 30
                    self.feeding = False
            else:
                self.feeding = False
            return

        target = [c for c in creatures if c.alive]
        if not target:
            return
        nearest = min(target, key=lambda c: distance(self.pos, c.pos))
        dist_to_prey = distance(self.pos, nearest.pos) / math.hypot(WIDTH, HEIGHT)
        angle_to_prey = angle_between(self.pos, nearest.pos)
        angle_diff = math.sin(angle_to_prey - self.angle)

        nearest_other = min(others, key=lambda p: distance(self.pos, p.pos)) if others else self
        dist_to_other = distance(self.pos, nearest_other.pos) / math.hypot(WIDTH, HEIGHT)

        inputs = np.array([angle_diff, dist_to_prey, dist_to_other, self.energy / 100])
        output = self.brain.forward(inputs)

        self.angle += output[0] * 0.2
        speed = max(0, output[1]) * 2
        self.pos[0] += math.cos(self.angle) * speed
        self.pos[1] += math.sin(self.angle) * speed
        self.pos[0] %= WIDTH
        self.pos[1] %= HEIGHT
        self.energy -= 0.15

        if distance(self.pos, nearest.pos) < 10 and nearest.alive:
            nearest.alive = False
            self.kills += 1
            self.feeding = True
            self.feed_timer = 0
            self.last_kill_pos = nearest.pos.copy()

        for other in others:
            if other is self or not other.alive:
                continue
            if distance(self.pos, other.pos) < 10 and self.energy < 15:
                if self.energy > other.energy:
                    other.alive = False
                    self.energy += 20
                elif self.energy < other.energy:
                    self.alive = False
                    other.energy += 20
                elif random.random() < 0.5:
                    other.alive = False
                    self.energy += 20
                else:
                    self.alive = False
                    other.energy += 20

    def draw(self):
        if not self.alive:
            return
        color = (255, 140, 0) if self.feeding else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), 8)

# === Simulation ===
class Simulation:
    def __init__(self):
        self.generation = 1
        self.creatures = [Creature() for _ in range(NUM_CREATURES)]
        self.predators = [Predator() for _ in range(NUM_PREDATORS)]
        self.foods = [Food() for _ in range(NUM_FOOD)]
        self.start_time = pygame.time.get_ticks()
        self.sim_speed = FPS

    def update(self):
        for predator in self.predators:
            predator.update(self.creatures, self.predators)
        for creature in self.creatures:
            creature.update(self.foods)

        if (pygame.time.get_ticks() - self.start_time) / 1000 > GENERATION_TIME:
            self.new_generation()

    def new_generation(self):
        top_creatures = sorted([c for c in self.creatures if c.alive], key=lambda c: c.score, reverse=True)[:NUM_CREATURES // 2]
        top_predators = sorted([p for p in self.predators if p.alive], key=lambda p: (p.kills, p.energy), reverse=True)[:NUM_PREDATORS // 2]

        if not top_creatures:
            top_creatures = [Creature() for _ in range(NUM_CREATURES // 2)]
        if not top_predators:
            top_predators = [Predator() for _ in range(NUM_PREDATORS // 2)]

        self.creatures = []
        for parent in top_creatures:
            child = Creature(parent.brain)
            child.brain.mutate()
            self.creatures.append(child)
            self.creatures.append(Creature(parent.brain))

        self.predators = []
        for parent in top_predators:
            child = Predator(parent.brain)
            child.brain.mutate()
            self.predators.append(child)
            self.predators.append(Predator(parent.brain))

        self.foods = [Food() for _ in range(NUM_FOOD)]
        self.start_time = pygame.time.get_ticks()
        self.generation += 1

    def draw(self):
        screen.fill((30, 30, 30))
        for food in self.foods:
            food.draw()
        for predator in self.predators:
            predator.draw()
        for creature in self.creatures:
            creature.draw()
        self.draw_ui()
        pygame.display.flip()

    def draw_ui(self):
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Gen: {self.generation} | Speed: {self.sim_speed} FPS | Creatures: {sum(c.alive for c in self.creatures)} | Predators: {sum(p.alive for p in self.predators)}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

# === Main Loop ===
sim = Simulation()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                sim.sim_speed = min(MAX_FPS, sim.sim_speed + 10)
            elif event.key == pygame.K_MINUS:
                sim.sim_speed = max(MIN_FPS, sim.sim_speed - 10)
            elif event.key == pygame.K_s:
                sim.new_generation()

    sim.update()
    sim.draw()
    clock.tick(sim.sim_speed)

pygame.quit()
sys.exit()
