import numpy as np
import matplotlib.pyplot as plt


class Rope:

    def __init__(self, n, m, g, k, rest_L, M, M_pos, anchor, dt, time, inits,
                 damping=0.3, moisture_content=0.0):

        self.dt = dt
        self.inits = inits
        self.timesteps = int(time / self.dt)
        self.moist = moisture_content
        self.damping = damping
        self.alpha = 1.0
        self.gamma = dt * 0.5

        rope_vector = M_pos - anchor
        distance = np.linalg.norm(rope_vector)
        if distance > rest_L:
            print("the distance between the anchor and the climber is larger than the length of rope, the rope is stretched")

        mx_pos = np.linspace(anchor[0], M_pos[0], n)
        my_pos = np.linspace(anchor[1], M_pos[1], n)
        self.pos = np.array([mx_pos, my_pos]).T

        self.masses = np.ones(n) * m / (n - 2) + self.moist * 0.4 * (np.ones(n) * m / (n - 2))
        self.masses[-1] = M

        self.v = np.zeros([n, 2])

        self.n = n
        self.g = g
        self.k = k
        self.rest = rest_L / (n - 1)
        self.M_pos = M_pos
        self.anchor = anchor

        self.f_hist = []
        self.p_hist = []
        self.v_hist = []

        self.f_hist.append(np.zeros((n, 2)))
        self.p_hist.append(self.pos)
        self.v_hist.append(self.v)

        self.equilibriate(self.inits)

    def equilibriate(self, timesteps):
        for i in range(timesteps):
            self.update()
            self.pos[-1] = self.M_pos
            self.v[-1] = np.zeros(2)
            self.f_hist.append(self.f)
            self.p_hist.append(self.pos)
            self.v_hist.append(self.v)
            if i % 100 == 0:
                print(i, end="\r")

    def run(self):
        for i in range(self.timesteps):
            step = i + self.inits
            self.update()
            self.f_hist.append(self.f)
            self.p_hist.append(self.pos)
            self.v_hist.append(self.v)

            if step % 100 == 0:
                print(step, end="\r")

            if np.linalg.norm(self.f[-1]) > 1e7:
                print(f"the rope has broken at timestep {step} due to a force of {np.linalg.norm(self.f[-1])}")
                break

    def forces(self, pos, velocities):
        self.f = np.zeros([self.n, 2])

        for i in range(1, self.n - 1):
            diff_Lx = pos[i, 0] - pos[i - 1, 0]
            diff_Ly = pos[i, 1] - pos[i - 1, 1]
            diff_Rx = pos[i, 0] - pos[i + 1, 0]
            diff_Ry = pos[i, 1] - pos[i + 1, 1]

            self.f[i, 0] = -self.k * (diff_Lx + diff_Rx) - self.damping * velocities[i, 0]
            self.f[i, 1] = -self.k * (diff_Ly + diff_Ry) - self.masses[i] * self.g - self.damping * velocities[i, 1]

        diff_Lx = pos[-1, 0] - pos[-2, 0]
        diff_Ly = pos[-1, 1] - pos[-2, 1]

        self.f[-1, 0] = -self.k * diff_Lx - self.damping * velocities[-1, 0]
        self.f[-1, 1] = -self.k * diff_Ly - self.masses[-1] * self.g - self.damping * velocities[-1, 1]

    def force_rest(self, pos, velocities):
        self.f = np.zeros([self.n, 2])

        for i in range(1, self.n - 1):

            left_vector = pos[i - 1] - pos[i]
            length_L = np.linalg.norm(left_vector)
            unit_left = left_vector / length_L if length_L != 0 else left_vector
            extension_L = length_L - self.rest

            if length_L < self.rest:
                K = self.k
            elif length_L < 1.4 * self.rest:
                K = self.k * (length_L / self.rest) ** 2 * self.alpha
            else:
                K = self.k * (length_L / self.rest) ** 2 * self.alpha + 1000 * (length_L / self.rest)

            force_left = -K * extension_L * unit_left
            self.f[i] += force_left
            self.f[i - 1] -= force_left

            right_vector = pos[i + 1] - pos[i]
            length_R = np.linalg.norm(right_vector)
            unit_right = right_vector / length_R if length_R != 0 else right_vector
            extension_R = length_R - self.rest

            if length_R < self.rest:
                K = self.k
            elif length_R < 1.4 * self.rest:
                K = self.k * (length_R / self.rest) ** 2 * self.alpha
            else:
                K = self.k * (length_R / self.rest) ** 2 * self.alpha + 1000 * (length_R / self.rest)

            force_right = -K * extension_R * unit_right
            self.f[i] += force_right
            self.f[i + 1] -= force_right

            v_rel = velocities[i] - velocities[i - 1]
            damping_force = -(self.k + self.moist) * np.dot(v_rel, unit_left) * unit_left
            gravity_force = np.array([0, -self.masses[i] * self.g])

            self.f[i] += damping_force + gravity_force

            if self.moist < 0.01:
                self.gamma = 0

            if length_L > self.rest:
                self.moist -= (self.gamma * (length_L / self.rest) ** 2) / self.n
            if length_R > self.rest:
                self.moist -= (self.gamma * (length_R / self.rest) ** 2) / self.n

        vector = pos[-2] - pos[-1]
        length = np.linalg.norm(vector)
        unit = vector / length if length != 0 else vector
        extension = length - self.rest

        force_hook = -self.k * extension * unit
        force_damping = -self.damping * velocities[-1]
        gravity_force = np.array([0, -self.masses[-1] * self.g])

        self.f[-1] = force_hook + force_damping + gravity_force

    def update(self):
        state = np.array([self.pos, self.v])
        new_state = self.rk4(state)
        self.pos = new_state[0]
        self.v = new_state[1]

    def derivatives(self, state):
        positions, velocities = state
        self.force_rest(positions, velocities)
        acceleration = self.f / self.masses[:, None]
        return np.array([velocities, acceleration])

    def rk4(self, state):
        k1 = self.dt * self.derivatives(state)
        k2 = self.dt * self.derivatives(state + 0.5 * k1)
        k3 = self.dt * self.derivatives(state + 0.5 * k2)
        k4 = self.dt * self.derivatives(state + k3)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def plot_force(forces, time, L, M):
    F = []
    for i in forces:
        avg = []
        for j in i:
            avg.append(np.linalg.norm(j))
        F.append(np.mean(avg))

    plt.title(f"Rope force, length = {L}, mass of climber = {M}")
    plt.plot(time, F)
    plt.show()


def main():
    segments = 50
    rope_weight = 5
    K = 5000
    length_of_rope = 10
    mass_of_climber = 75

    anchor = np.zeros(2)
    climber_position = np.zeros(2)
    climber_position[0] = -8

    rope = Rope(
        segments, rope_weight, 9.81, K,
        length_of_rope, mass_of_climber,
        climber_position, anchor,
        0.001, 20, 1000
    )

    rope.run()

    t = rope.timesteps + rope.inits
    x = np.linspace(0, t * rope.dt, t + 1)

    y = np.array(rope.p_hist)[:, -1, 1]

    plt.plot(x, y)
    plt.axhline(0, color="r", linestyle="--")
    plt.show()

    plot_force(rope.f_hist, x, length_of_rope, mass_of_climber)


if __name__ == "__main__":
    main()

