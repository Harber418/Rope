import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Rope:

    def __init__(self, N, m, g, k, rest_L, M, M_pos, anchor, dt, time,
                 damping=0, moisture_content=0.0, air_resistance=0, theta=90.0):

        n = N + 2 # number of masses plus the climber and the anchor
        self.dt = dt
        self.timesteps = int(time / self.dt)
        self.moist = moisture_content
        self.damping = damping
        self.air_resistance = air_resistance
        self.alpha = 1.0
        self.gamma = dt * 0.5

        #this doesnt mean we cant run the simulation just that the rope is streached from the start
        if np.linalg.norm(M_pos - anchor) > rest_L:
            print("the distance between the anchor and the climber is larger than the length of rope, the rope is stretched")
        self.theta=theta
        
        if theta == 90.0:
            angle= False
        else:
            angle = True 
        self.angle=angle
        #i think this is a mistake but perhaps it is correct not too sure # what is a mistake?
        mx_pos = np.linspace(anchor[0], M_pos[0], n)
        my_pos = np.linspace(anchor[1], M_pos[1], n)
        self.pos = np.array([mx_pos, my_pos]).T

        self.masses = np.ones(n) * m / N + self.moist * 0.4 * (np.ones(n) * m / N)
        self.masses[-1] = M

        self.v = np.zeros([n, 2])

        self.n = n
        self.g = g
        self.k = k
        self.l0 = rest_L / (N + 1)  # N+1 springs between N+2 masses
        self.M_pos = M_pos
        self.anchor = anchor

        self.f_hist = []
        self.p_hist = []
        self.v_hist = []

        self.f_hist.append(np.zeros((n, 2)))
        self.p_hist.append(self.pos)
        self.v_hist.append(self.v)

    def run(self):
        for i in range(self.timesteps):
            self.update()
            self.f_hist.append(self.f)
            self.p_hist.append(self.pos.copy())
            self.v_hist.append(self.v.copy())

            if i % 100 == 0:
                print(i, end="\r")

    def spring_force(self, ri, rj, vi, vj):
        dx = rj - ri
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

        # must start at anchor to find L force on first mass
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
        new_state = self.rk4(state)
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

        if tracker:
            self.f = forces

        return np.array([velocities, acceleration])

    def rk4(self, state):
        k1 = self.dt * self.derivatives(state, True)
        k2 = self.dt * self.derivatives(state + 0.5 * k1, False)
        k3 = self.dt * self.derivatives(state + 0.5 * k2, False)
        k4 = self.dt * self.derivatives(state + k3, False)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
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
        plt.title('Total Kinetic Energy vs Time')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def wall(self):
    	"""if the climber hits the wall thier should be a change of momentum
    	I think keeping this just for the climber and not the whole rope is good enough, 
    	may cost too much to do it for the whole rope"""
    	if self.angle:
    		#find if the climber is in the wall
    		ywall = (self.pos[-1,0]-self.anchor[0])*np.sin(self.theta) + self.anchor[1]
    		if self.pos[-1,1] > ywall:
    			self.v[-1] = - self.v[-1]
    			
    	#could add the wall into the animation as a static line. probably just generate some values and plot.
    			
    def scatter_position(self):
    	"""plots the postion of the climber as time goes on, I think this should be animated"""
    	t = rope.timesteps
   	time = np.linspace(0, t * rope.dt, t + 1)
   	x = np.array(rope.p_hist)[:,-1,0]

    	y = np.array(rope.p_hist)[:, -1, 1]
    	
    	plt.scatter(x,y)
    	plt.show()
    	
    def gif(self):
    	"""a 2D animation for the climber falling,"""
    	fig, ax = plt.subplots()
    	artists =[]
    	colors = ['tab:blue', 'tab:purple','tab:red']
    	for i in range(self.timesteps):
    		x = np.array(rope.p_hist)[i,-1,0]
    		y = np.array(rope.p_hist)[i,-1, 1]
    		container = ax.scatter(x,y, color='b')
    		artists.append(container)
    	ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=10)
    	plt.show()
    	
    def fall_factor_normal(self):

    	
    def Fall_factor_calc(self):
    	#idea for what fall factor could be 
    	#the ratio of max hight - min height to the anchor height - min height
    	low_point = np.min(self.p_hist[:,-1,1])
    	initial_fall_height =(self.p_hist[0,-1,1])
    	total_fall = initial_fall_height -low_point
    	streached_rope_length = np.abs(self.anchor[1] - low_point)
    	
    	fall_factor = total_fall/streached_rope_length
    	#idk what this is but i don't want to get rid of it uncase its useful lol
	#rope_vector = self.M_pos - self.anchor
	#low_point = np.array([self.anchor[0], (self.anchor[1]-self.rest_L)])
	#fall_factor = rope_vector[1]/np.linalg.norm(low_point)
	return fall_factor
    
def main(segments, rope_weight, K, length_of_rope, mass_of_climber, climber_position, time, damping, moisture, air_resistance=0):

    anchor = np.zeros(2)

    rope = Rope(
        segments, rope_weight, 9.81, K,
        length_of_rope, mass_of_climber,
        climber_position, anchor,
        0.001, time, damping, moisture, air_resistance  # Reduced timestep
    )

    rope.run()

    t = rope.timesteps
    x = np.linspace(0, t * rope.dt, t + 1)

    y = np.array(rope.p_hist)[:, -1, 1]

    plt.plot(x, y)
    plt.axhline(0, color="r", linestyle="--")
    plt.show()

    rope.plot_kinetic_energy()
    #rope.scatter_position()
    rope.gif()
    

    
if __name__ == "__main__":
    main(50, 5, 20000, 10, 75, np.array([0, 10]), 40, 20, 0, 1)


