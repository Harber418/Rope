import numpy as np
import matplotlib.pyplot as plt


class Rope:


    def __init__(self, n, m, g, k, rest_L, M, M_pos, anchor, dt, time, inits, damping=0.0):
        self.dt = dt
        self.inits = inits
        self.timesteps = int(time / self.dt)
        #"normalised between 0 and 1"
        self.damping = damping
        
        mx_pos = np.linspace(anchor[0], M_pos[0], n)
        my_pos = np.linspace(anchor[1], M_pos[1], n)
        self.pos = np.array([mx_pos, my_pos]).T
        
        self.masses = np.ones(n) * m / (n-2)
        self.masses[-1] = M      

        self.v = np.zeros([n, 2])

        self.n = n
        self.g = g
        self.k = k
        self.rest = rest_L / n
        self.M_pos = M_pos
        self.anchor = anchor

        self.f_hist = []
        self.p_hist = []
        self.v_hist = []

        self.f_hist.append(np.zeros(n))
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
            if i // 100 == i / 100:
                print(i, end='\r')

    def run(self):
        for i in range(self.timesteps):
            i += self.inits
            self.update()
            self.f_hist.append(self.f)
            self.p_hist.append(self.pos)
            self.v_hist.append(self.v)
            if i // 100 == i / 100:
                print(i, end='\r')
        

    def forces(self, pos, velocities):
        self.f = np.zeros([self.n, 2])

        # force on 0th mass is always 0; anchor point
        for i in range(1, self.n-1):
            diff_Lx = pos[i, 0] - pos[i-1, 0]
            diff_Ly = pos[i, 1] - pos[i-1, 1]
            diff_Rx = pos[i, 0] - pos[i+1, 0]
            diff_Ry = pos[i, 1] - pos[i+1, 1]

	    #update forces for spring in the x axis 
            self.f[i, 0] = - self.k * (diff_Lx - np.sign(diff_Lx) * self.rest + diff_Rx - np.sign(diff_Rx) * self.rest) - self.damping*velocities[i,0]
            #update forces for sping in y axis 
            self.f[i, 1] = - self.k * (diff_Ly - np.sign(diff_Ly) * self.rest + diff_Ry - np.sign(diff_Ry) * self.rest) - self.masses[i] * self.g - self.damping*velocities[i,1]

	#update forces for climber (mass M )
        diff_Lx = pos[-1, 0] - pos[-2, 0]
        diff_Ly = pos[-1, 1] - pos[-2, 1]

        self.f[-1, 0] = - self.k * (diff_Lx - np.sign(diff_Lx) * self.rest) - self.damping*velocities[-1,0] 
        self.f[-1, 1] = - self.k * (diff_Ly - np.sign(diff_Ly) * self.rest) - self.masses[-1] * self.g -self.damping*velocities[-1,1]

    def update(self):
        state = np.array([self.pos, self.v])
        new_state = self.rk4(state)
        self.pos = new_state[0]
        self.v = new_state[1]

        
        
    def derivatives(self, state):
        positions = state[0]
        velocities = state[1]
        
        #"updates self.f with spring and gravity forces and damping"
        self.forces(positions, velocities)
        #acceleration is now based on force of gravity, spring and damping"
        acceleration = self.f / self.masses[:, None]
        
        dpositions = velocities
        dvelocities = acceleration
        
        return np.array([dpositions, dvelocities])

    def rk4(self, state):
        #rk4 is v cool
        k1 = self.dt * self.derivatives(state)
        k2 = self.dt * self.derivatives(state + 0.5 * k1)
        k3 = self.dt * self.derivatives(state + 0.5 * k2)
        k4 = self.dt * self.derivatives(state + k3)
        
        return state + (1/6)*(k1 + 2*k2 + 2*k3 + k4)

def main():
  rope = Rope(50, 5, 9.81, 5000, 10, 75, np.zeros(2), np.zeros(2), 0.001, 30, 0,)
  rope.run()
  "plot"
  t = rope.timesteps + rope.inits
  dt = rope.dt
  x = np.linspace(0, t*dt, t + 1)
  y0 = np.array(rope.p_hist)
  y = y0[:, -1]
  y = y[0:, 1]
  
  plt.plot(x, y)
  plt.axhline(0, color = 'r', linestyle='--')
  plt.show()


if __name__ == "__main__":
    main()
