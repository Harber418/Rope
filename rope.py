import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Rope:

    def __init__(self, N, m, g, k, rest_L, M, M_pos, anchor, dt, time,
                 damping=0, moisture_content=0.0, air_resistance=0, 
                 theta=80.0, angle = True):
        

        n = N + 2 # number of masses plus the climber and the anchor
        self.dt = dt
        self.timesteps = int(time / self.dt)
        self.moist = moisture_content
        self.damping = damping
        self.air_resistance = air_resistance

        if np.linalg.norm(M_pos - anchor) > rest_L:
            print("the distance between the anchor and the climber is larger than the length of rope, the rope is stretched")
        
        self.theta=theta
        self.angle=angle
        
        mx_pos = np.linspace(anchor[0], M_pos[0], n)
        my_pos = np.linspace(anchor[1], M_pos[1], n)

        self.pos = np.array([mx_pos, my_pos]).T

        self.masses = np.ones(n) * m / N + self.moist * 0.4 * (np.ones(n) * m / N)
        self.masses[-1] = M

        self.v = np.zeros([n, 2])

        self.n = n
        self.g = g
        self.k = k
        self.rest = rest_L
        self.l0 = rest_L / (N + 1)  # N+1 springs between N+2 masses
        self.M_pos = M_pos
        self.anchor = anchor

        self.f_hist = []
        self.p_hist = []
        self.v_hist = []

        self.f_hist.append(np.zeros((n, 2)))
        self.p_hist.append(self.pos)
        self.v_hist.append(self.v)

        self.run(5) # allow the rope to equilibriate before the fall

    def run(self, t):
        """THIS IS THE EQUILIBRIATION FUNCTION"""
        timesteps = t / self.dt
        print(f"Equilibriating, please wait {t} seconds.")

        for i in range(int(timesteps)):
            self.update()
            self.pos[-1] = self.M_pos
            self.v[-1] = np.zeros(2)

        print("Equilibriation complete!")

    
    def run_with_live_animation(self, update_interval=50):
        """Run the simulation with live animation of the climber's position"""
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot limits based on initial conditions
        y_min = min(self.anchor[1], self.M_pos[1]) - self.l0 * (self.n - 1) * 2
        y_max = max(self.anchor[1], self.M_pos[1]) + 2
        x_range = abs(self.M_pos[0] - self.anchor[0]) + 5
        ax.set_xlim(self.anchor[0] - x_range, self.anchor[0] + x_range)
        ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Live Rope Simulation (units: m, s, kg)')
        ax.grid(True, alpha=0.3)
        ax.axhline(self.anchor[1], color='gray', linestyle='--', alpha=0.5, label='Anchor Level (m)')
        
        # Initialize plot elements
        rope_line, = ax.plot([], [], 'b-', linewidth=2, label='Rope')
        climber_point, = ax.plot([], [], 'ro', markersize=10, label='Climber')
        anchor_point, = ax.plot(self.anchor[0], self.anchor[1], 'ks', markersize=12, label='Anchor')
        
        # Plot the wall if angle is enabled
        if self.angle:
            wall_x, wall_y = self.wall()
            ax.plot(wall_x, wall_y, color='k', label='wall')

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right')
        
        for i in range(self.timesteps):
            self.update()
            if self.angle:
                self.wall()
            self.f_hist.append(self.f)
            self.p_hist.append(self.pos.copy())
            self.v_hist.append(self.v.copy())
            
            # Update the plot at specified intervals
            if i % update_interval == 0:
                # Update rope line
                rope_line.set_data(self.pos[:, 0], self.pos[:, 1])
                
                # Update climber position
                climber_point.set_data([self.pos[-1, 0]], [self.pos[-1, 1]])
                
                # Update time text
                current_time = i * self.dt
                time_text.set_text(f'Time: {current_time:.3f}s\nStep: {i}/{self.timesteps}')
                
                # Redraw the plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
            
            if i % 1000 == 0:
                print(f"Progress: {i}/{self.timesteps}", end="\r")
        
        plt.ioff()  # Turn off interactive mode
        print(f"\nSimulation complete!")
        plt.show()

    def spring_force(self, ri, rj, vi, vj):
        dx = rj - ri
        # spring length
        l = np.linalg.norm(dx)

        if l == 0:
            return np.zeros(2), np.zeros(2)
        # rope does not resist compression
        elif l <= self.l0:
            return np.zeros(2), np.zeros(2)

        # unit vector along spring
        n_hat = dx / l

        # elastic force magnitude (Hooke)
        Fs_mag = self.k * (l - self.l0)
        F_elastic = Fs_mag * n_hat

        # relative velocity along spring direction
        v_rel = np.dot(vj - vi, n_hat)

        # damping force on mass i (opposes relative motion)
        F_damping = self.damping * v_rel * n_hat

        # return elastic and damping forces on mass i
        return F_elastic, F_damping


            
    def forces(self, pos, velocities):
        f = np.zeros((self.n, 2))

        # must start at anchor to find force due to 
        # anchor spring on first mass
        for i in range(self.n - 1):
            F_e, F_d = self.spring_force(
                pos[i],
                pos[i + 1],
                velocities[i],
                velocities[i + 1]
            )

            # Newton's third law 
            f[i] += (F_e + F_d)
            f[i + 1] -= (F_e + F_d)
        

        # gravity
        for i in range(self.n):
            f[i, 1] -= self.masses[i] * self.g
            
            # air resistance (drag proportional to velocity)
            f[i] -= self.air_resistance * velocities[i]

        return f


    def update(self):
        state = np.array([self.pos, self.v])
        new_state = self.method(state)
        self.pos = new_state[0]
        self.v = new_state[1]
        
        # Enforce anchor constraint after integration
        self.pos[0] = self.anchor
        self.v[0] = np.zeros(2)

        # Calculate self.f once for history tracking
        self.f = self.forces(self.pos, self.v)

    def derivatives(self, state, tracker):
        positions, velocities = state
        forces = self.forces(positions, velocities)
        acceleration = forces / self.masses[:, None]

        # store forces if operating on the actual state
        if tracker:
            self.f = forces

        # derivative of [position, velocity] is [velocity, acceleration]
        return np.array([velocities, acceleration])

    def method(self, u_n):
        # RK4 method for updating the state of the system
        U1 = self.dt * self.derivatives(u_n, True)
        U2 = self.dt * self.derivatives(u_n + 0.5 * U1, False)
        U3 = self.dt * self.derivatives(u_n + 0.5 * U2, False)
        U4 = self.dt * self.derivatives(u_n + U3, False)
        # return u_n+1
        return u_n + (U1 + 2 * U2 + 2 * U3 + U4) / 6

    def plot_kinetic_energy(self):
        """Plot total kinetic energy of the system over time"""
        KE_total = []
        
        for velocities in self.v_hist:
            # Calculate KE for each mass: 0.5 * m * v^2
            v_squared = np.sum(velocities**2, axis=1)  # |v|^2 for each mass
            KE = 0.5 * self.masses * v_squared
            KE_total.append(np.sum(KE))
        
        time = np.linspace(0, self.timesteps * self.dt, len(KE_total))
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, KE_total, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Total Kinetic Energy (J)')
        plt.title('Total Kinetic Energy vs Time (units: J, s, kg, m)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def rope_force(self):
        """a function to plot the average force in the rope at each step in the simulation"""
        avg_forces = []
        total_forces = []
        for forces in self.f_hist:
            # no anchor no climber 
            rope_forces = forces[1:-1]
            # axis=1 is used as forces are in 2D
            norms = np.linalg.norm(rope_forces, axis=1)
            avg_force = np.mean(norms)
            total_force = np.sum(norms)
            total_forces.append(total_force)
            avg_forces.append(avg_force)

        time_average = np.arange(len(avg_forces)) * self.dt
        time_total = np.arange(len(total_forces)) * self.dt
        smooth = True 
        # the switch to make th force graph look smooth, this helps see the trend better without fluctuations 
        if smooth:
            smoothest_forces = []
            smooth_forces = []
            for i in range(len(avg_forces)):
                smooth_forces.append(total_forces[i])
                if i % 10 == 0:
                    # do something every 3 steps
                    mens = np.mean(smooth_forces)
                    smoothest_forces.append(mens)
                    smooth_forces = []
            total_forces = smoothest_forces
            time_total = np.arange(len(total_forces)) * self.dt * 10

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        axs[0].plot(time_average, avg_forces, linewidth=1.5)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Average Rope Force (N)')
        axs[0].set_title('Average Rope Force vs Time (units: N, s)')
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(time_total, total_forces, color='tab:orange', linewidth=1.5)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Total Rope Force (N)')
        axs[1].set_title('Total Rope Force vs Time (units: N, s)')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def wall(self):
        """Reflect the climber's velocity if they hit the wall (momentum change)."""
        # Wall: y = tan(theta) * (x - anchor_x) + anchor_y
        theta_rad = np.deg2rad(self.theta) if self.theta > 2 * np.pi else self.theta
        x_c, y_c = self.pos[-1, 0], self.pos[-1, 1]
        ywall = np.tan(theta_rad) * (x_c - self.anchor[0]) + self.anchor[1]

        # Wall normal vector (perpendicular to wall)
        n = np.array([-np.sin(theta_rad), np.cos(theta_rad)])
        v = self.v[-1]

        # Only reflect if the climber is past the wall and moving into the wall
        if y_c > ywall and np.dot(v, n) > 0:
            v_n = np.dot(v, n) * n  # normal component
            v_t = v - v_n           # tangential component
            collision_damping = 0.15       # 1.0 = elastic, <1.0 = inelastic
            self.v[-1] = v_t - collision_damping * v_n
            print(f"the climber hit the wall and damping factor is {collision_damping}")

        wall_x = np.array([self.anchor[0]-10 ,self.anchor[0], self.anchor[0] + 10])
        wall_y = np.array([self.anchor[1] - 10*np.tan(theta_rad),self.anchor[1], self.anchor[1] + 10*np.tan(theta_rad)])
        
        return (wall_x, wall_y)
    
    def fall(self):
        return (self.M_pos[1]+(self.anchor[1]-self.rest))/self.rest

        
    def Fall_factor_calc(self):
        #idea for what fall factor could be 
        #the ratio of max hight - min height to the anchor height - min height
        low_point = np.min(self.p_hist[:,-1,1])
        initial_fall_height =(self.p_hist[0,-1,1])
        total_fall = initial_fall_height -low_point
        stretched_rope_length = np.abs(self.anchor[1] - low_point)
        
        fall_factor = total_fall/stretched_rope_length
        #idk what this is but i don't want to get rid of it uncase its useful lol
        #rope_vector = self.M_pos - self.anchor
        #low_point = np.array([self.anchor[0], (self.anchor[1]-self.rest_L)])
        #fall_factor = rope_vector[1]/np.linalg.norm(low_point)
        return fall_factor
    
    def save_history(self, filename, fall_factor):
        """Save p_hist, f_hist, v_hist, and fall_factor to a .npz file for later analysis."""
        np.savez(filename, p_hist=np.array(self.p_hist), f_hist=np.array(self.f_hist), v_hist=np.array(self.v_hist), fall_factor=fall_factor)
    

def main(segments, rope_weight, K, length_of_rope, mass_of_climber, climber_position, time, dt, damping, moisture, air_resistance=0):

    anchor = np.zeros(2)

    rope = Rope(
        segments, rope_weight, 9.81, K,
        length_of_rope, mass_of_climber,
        climber_position, anchor,
        dt, time, damping, moisture, air_resistance  # Reduced timestep
    )

    rope.run_with_live_animation(update_interval=50)

    t = rope.timesteps
    x = np.linspace(0, t * rope.dt, t + 1)

    y = np.array(rope.p_hist)[:, -1, 1]

    plt.plot(x, y)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel('Time (s)')
    plt.ylabel('Climber Y Position (m)')
    plt.title('Climber Y Position vs Time (units: m, s, kg)')
    plt.show()

    rope.plot_kinetic_energy()
  
    rope.rope_force()
    #ff = rope.Fall_factor_calc()
    #print(f"fall factor is {ff}")
    fall_factor = rope.fall()
    print(fall_factor)
    
    rope.save_history("rope_simulation_data_rk4.npz", fall_factor)

if __name__ == "__main__":
    main(30, 5, 40000, 10, 75, np.array([5, 0]), 30, 0.001, 30, 0, 0.3)


