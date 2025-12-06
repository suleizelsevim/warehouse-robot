import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# =====================================================
#             PARAMETRELER (ÖNEMLİ KISIM)
# =====================================================

GRID_SIZE = 6

STEP_PENALTY = -0.1
PICKUP_REWARD = 20.0
DELIVERY_REWARD = 50.0

EPISODES = 150000
MAX_STEPS = 500

ALPHA = 0.1
GAMMA = 0.99
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY_EPISODES = EPISODES * 0.9

# =====================================================
#                   ORTAM SINIFI
# =====================================================

class WarehouseEnv:
    def __init__(self):
        self.grid = GRID_SIZE
        self.start = (0, 0)
        self.exit = (GRID_SIZE - 1, GRID_SIZE - 1)

        self.pallets = None
        self.collected = None
        self.robot_pos = None

    def reset(self):
        self.robot_pos = self.start

        # 3 adet rastgele palet – çıkış ve start dışında
        forbidden = {self.start, self.exit}
        cells = [(r, c) for r in range(self.grid) for c in range(self.grid)
                 if (r, c) not in forbidden]

        self.pallets = tuple(sorted(random.sample(cells, 3)))
        self.collected = (False, False, False)

        return self.encode_state()

    # -------------------------------------------------

    def encode_state(self):
        # State = (robot_r, robot_c, pallet_locations, collected_mask)
        mask = 0
        for i, c in enumerate(self.collected):
            if c:
                mask |= (1 << i)

        return (self.robot_pos[0], self.robot_pos[1], self.pallets, mask)

    # -------------------------------------------------

    def is_done(self):
        return all(self.collected) and self.robot_pos == self.exit

    # -------------------------------------------------

    def step(self, action):
        """
        Actions:
            0 = UP
            1 = DOWN
            2 = LEFT
            3 = RIGHT
        """
        r, c = self.robot_pos

        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and r < self.grid - 1:
            r += 1
        elif action == 2 and c > 0:
            c -= 1
        elif action == 3 and c < self.grid - 1:
            c += 1

        self.robot_pos = (r, c)
        reward = STEP_PENALTY

        # =====================================================
        #                PALET TOPLAMA ÖDÜLÜ
        # =====================================================
        collected_list = list(self.collected)
        for i, loc in enumerate(self.pallets):
            if not collected_list[i] and self.robot_pos == loc:
                collected_list[i] = True
                reward += PICKUP_REWARD

        self.collected = tuple(collected_list)

        # =====================================================
        #                ÇIKIŞ ÖDÜLÜ
        # =====================================================
        done = False
        if self.is_done():
            reward += DELIVERY_REWARD
            done = True

        return self.encode_state(), reward, done


# =====================================================
#               Q-LEARNING (DICT TABLE)
# =====================================================

def train():
    env = WarehouseEnv()
    q = {}

    epsilon = EPS_START
    episode_rewards = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            # ACTION SEÇİMİ
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                if state not in q:
                    q[state] = np.zeros(4)
                action = int(np.argmax(q[state]))

            next_state, reward, done = env.step(action)

            # Q UPDATE
            if state not in q:
                q[state] = np.zeros(4)
            if next_state not in q:
                q[next_state] = np.zeros(4)

            q[state][action] = (1 - ALPHA) * q[state][action] + \
                               ALPHA * (reward + GAMMA * np.max(q[next_state]))

            state = next_state
            total_reward += reward

            if done:
                break

        # EPSILON DECAY
        if ep < EPS_DECAY_EPISODES:
            epsilon = EPS_START - (EPS_START - EPS_MIN) * (ep / EPS_DECAY_EPISODES)
        else:
            epsilon = EPS_MIN

        episode_rewards.append(total_reward)

        if ep % 10000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {ep}/{EPISODES} avg_reward(last 1000)={avg_reward:.1f} epsilon={epsilon:.3f}")

    print("\nTraining finished.\n")
    return q, episode_rewards


# =====================================================
#                DEĞERLENDİRME
# =====================================================

def evaluate(q, episodes=50):
    env = WarehouseEnv()
    success = 0
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            if state in q:
                action = int(np.argmax(q[state]))
            else:
                action = random.randint(0, 3)

            state, reward, done = env.step(action)
            total_reward += reward

            if done:
                success += 1
                break

        rewards.append(total_reward)

    print("=== EVALUATION RESULTS ===")
    print("Episodes       :", episodes)
    print("Success count  :", success)
    print("Success rate   :", f"{success / episodes * 100:.1f}%")
    print("Avg reward     :", f"{np.mean(rewards):.2f}")


# =====================================================
#                  ANİMASYON (GIF)
# =====================================================

def _render_env(env, ax, ep, step, total_reward):
    ax.clear()

    # Grid hücreleri
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            facecolor = "white"

            if (r, c) == env.start:
                facecolor = "lightblue"   # başlangıç
            if (r, c) == env.exit:
                facecolor = "lightpink"  # çıkış

            # palet var mı?
            if env.pallets is not None and (r, c) in env.pallets:
                idx = env.pallets.index((r, c))
                if env.collected[idx]:
                    facecolor = "palegreen"   # toplanmış palet
                else:
                    facecolor = "salmon"      # toplanmamış palet

            # (0,0) yukarıda olsun diye çizim sırasında satırları ters çeviriyoruz
            rect = patches.Rectangle(
                (c, GRID_SIZE - 1 - r), 1, 1,
                edgecolor="black",
                facecolor=facecolor,
                linewidth=1
            )
            ax.add_patch(rect)

    # Robotu çiz
    rr, rc = env.robot_pos
    ax.add_patch(
        patches.Rectangle(
            (rc, GRID_SIZE - 1 - rr), 1, 1,
            fill=False,
            edgecolor="gold",
            linewidth=3
        )
    )

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"Episode {ep} | Step {step}\nTotal reward = {total_reward:.1f}",
        fontsize=10
    )


def save_gif(q, filename="warehouse.gif", episodes=3, max_steps=200, delay=0.15):
    env = WarehouseEnv()
    frames = []

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        step = 0

        done = False
        while not done and step < max_steps:
            # Çizim
            _render_env(env, ax, ep, step, total_reward)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = Image.fromarray(rgba)
            frames.append(frame.convert("P", palette=Image.ADAPTIVE))

            # Aksiyon (greedy)
            if state in q:
                action = int(np.argmax(q[state]))
            else:
                action = random.randint(0, 3)

            state, reward, done = env.step(action)
            total_reward += reward
            step += 1

    plt.close(fig)

    if not frames:
        print("Frame yok, GIF oluşturulamadı.")
        return

    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=int(delay * 1000),
        loop=0
    )
    print(f"GIF kaydedildi: {filename}")


# =====================================================
#                      MAIN
# =====================================================

if __name__ == "__main__":
    q, rewards = train()
    evaluate(q, episodes=100)

    # Eğitilmiş ajan için GIF üret
    save_gif(q, filename="warehouse.gif", episodes=20, max_steps=40, delay=0.3)
