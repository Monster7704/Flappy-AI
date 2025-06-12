import pygame
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH = 400
HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)

# Bird settings
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
GRAVITY = 0.5
FLAP_STRENGTH = -10

# Pipe settings
PIPE_WIDTH = 70
PIPE_GAP = 150

# RL Settings
INPUT_DIM = 5
OUTPUT_DIM = 2
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001

# Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Helper functions
def get_state(bird, pipes):
    if not pipes:
        return [0] * INPUT_DIM

    pipe = pipes[0]
    state = [
        bird.y / HEIGHT,
        pipe.height / HEIGHT,
        (pipe.x - bird.x) / WIDTH,
        (pipe.height + PIPE_GAP - bird.y) / HEIGHT,
        bird.velocity / 10
    ]
    return np.array(state, dtype=np.float32)

def get_reward(bird, pipes, done):
    if done:
        return -1000
    reward = 0.1
    for pipe in pipes:
        if pipe.x + PIPE_WIDTH < bird.x and not hasattr(pipe, 'passed'):
            pipe.passed = True
            reward += 10
    return reward

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), device=device),
            torch.tensor(actions, device=device),
            torch.tensor(rewards, device=device),
            torch.tensor(np.array(next_states), device=device),
            torch.tensor(dones, device=device)
        )

    def __len__(self):
        return len(self.buffer)

# Load Assets
BACKGROUND_IMG = pygame.image.load("assets/background-day.png").convert()
GROUND_IMG = pygame.image.load("assets/base.png").convert()
PIPE_IMG = pygame.image.load("assets/pipe-green.png").convert_alpha()

BIRD_IMGS = [
    pygame.image.load("assets/yellowbird-upflap.png").convert_alpha(),
    pygame.image.load("assets/yellowbird-midflap.png").convert_alpha(),
    pygame.image.load("assets/yellowbird-downflap.png").convert_alpha()
]

# Bird Class
class Bird:
    IMGS = BIRD_IMGS
    ANIMATION_TIME = 5

    def __init__(self):
        self.x = 50
        self.y = HEIGHT // 2
        self.velocity = 0
        self.img_index = 0
        self.img = self.IMGS[self.img_index]
        self.rect = self.img.get_rect(center=(self.x, self.y))

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.centery = int(self.y)

        # Animate wings
        self.img_index = (pygame.time.get_ticks() // self.ANIMATION_TIME) % len(self.IMGS)
        self.img = self.IMGS[self.img_index]

    def draw(self):
        screen.blit(self.img, self.rect.topleft)

# Pipe Class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, HEIGHT - PIPE_GAP - 100)
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        self.bottom_rect = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT - self.height - PIPE_GAP)

    def update(self, speed):
        self.x -= speed
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self):
        screen.blit(PIPE_IMG, (self.x, 0))  # Top pipe
        flip_pipe = pygame.transform.flip(PIPE_IMG, False, True)
        screen.blit(flip_pipe, (self.x, self.height + PIPE_GAP))  # Bottom pipe

# Draw Game Window
def draw_window(bird, pipes, score, generation):
    screen.blit(BACKGROUND_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw()
    screen.blit(GROUND_IMG, (0, HEIGHT - GROUND_IMG.get_height()))
    bird.draw()

    # Score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Generation counter (top right)
    gen_text = font.render(f"Gen: {generation}", True, WHITE)
    screen.blit(gen_text, (WIDTH - gen_text.get_width() - 10, 10))

# Show Game Over Screen
def show_game_over_screen(score):
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))

    game_over_text = font.render("GAME OVER", True, WHITE)
    score_text = font.render(f"Score: {score}", True, WHITE)
    restart_text = font.render("Press R to Restart", True, WHITE)

    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 60))
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, HEIGHT // 2))
    screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT // 2 + 40))

# Main Menu
def main_menu():
    selected = 0
    options = ["AI Play", "Play", "Exit"]
    moving = False

    while True:
        clock.tick(FPS)
        screen.blit(BACKGROUND_IMG, (0, 0))
        title_text = font.render("Flappy Bird AI", True, WHITE)
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 50))

        for i, option in enumerate(options):
            color = (255, 215, 0) if i == selected else WHITE
            text = font.render(option, True, color)
            rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + i * 50 - 40))
            screen.blit(text, rect)

        pygame.display.flip()

        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if keys[pygame.K_DOWN] and not moving:
            selected = (selected + 1) % len(options)
            moving = True
        elif keys[pygame.K_UP] and not moving:
            selected = (selected - 1) % len(options)
            moving = True
        elif keys[pygame.K_RETURN]:
            if options[selected] == "Exit":
                pygame.quit()
                sys.exit()
            elif options[selected] == "AI Play":
                return False  # AI mode
            elif options[selected] == "Play":
                return True   # Manual mode

        if not keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            moving = False

# Training step
def train_step(model, target_model, buffer, optimizer):
    model.train()
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        expected_q_values = rewards + (GAMMA * next_q_values * (~dones))

    loss = torch.nn.functional.mse_loss(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main function
def main():
    global MANUAL_MODE

    is_manual = main_menu()
    MANUAL_MODE = is_manual

    model = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    target_model = DQN(INPUT_DIM, OUTPUT_DIM).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = 1.0
    episode = 0
    best_score = 0

    while True:
        bird = Bird()
        pipes = [Pipe(WIDTH)]
        score = 0
        speed = 3
        done = False
        game_over = False

        state = get_state(bird, pipes)

        while not done:
            clock.tick(FPS)
            screen.fill((135, 206, 250))

            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = 0
            if MANUAL_MODE:
                if keys[pygame.K_SPACE]:
                    bird.flap()
            else:
                if np.random.rand() < epsilon:
                    action = np.random.choice([0, 1])
                else:
                    with torch.no_grad():
                        q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
                        action = q_values.argmax().item()
                if action == 1:
                    bird.flap()

            bird.update()
            for pipe in pipes:
                pipe.update(speed)

            if pipes[-1].x < WIDTH - 200:
                pipes.append(Pipe(WIDTH))
            if pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                score += 1

            # Collision detection
            if bird.y > HEIGHT - GROUND_IMG.get_height() or bird.y < 0:
                done = True
                game_over = True

            for pipe in pipes:
                if bird.rect.colliderect(pipe.top_rect) or bird.rect.colliderect(pipe.bottom_rect):
                    done = True
                    game_over = True

            next_state = get_state(bird, pipes)
            reward = get_reward(bird, pipes, done)
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= BATCH_SIZE:
                train_step(model, target_model, buffer, optimizer)

            draw_window(bird, pipes, score, episode)
            if game_over:
                show_game_over_screen(score)

            if game_over and keys[pygame.K_r]:
                done = True

            pygame.display.flip()

        print(f"Episode: {episode} | Score: {score} | Epsilon: {epsilon:.3f}")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "best_flappy_dqn.pth")
            print("New best model saved!")

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        episode += 1

if __name__ == "__main__":
    main()