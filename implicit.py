from rope import Rope
from rope import main
import numpy as np
from matplotlib import pyplot as plt

class ImplicitRope(Rope):

    def method(self, state):
        """use the 2-stage Gauss-Legendre method to update the state"""
        positions, velocities = state
        
        # 2-stage Gauss-Legendre Butcher tableau coefficients
        sqrt3 = np.sqrt(3)
        a11 = 0.25
        a12 = 0.25 - sqrt3 / 6.0
        a21 = 0.25 + sqrt3 / 6.0
        a22 = 0.25
        b1 = 0.5
        b2 = 0.5
        
        # Initial guess for k1 and k2 using explicit Euler
        k1 = self.derivatives(state, True)
        k2 = k1.copy()
        
        # Fixed-point iteration to solve the implicit system
        max_iterations = 10
        tolerance = 1e-10
        
        for iteration in range(max_iterations):
            k1_old = k1.copy()
            k2_old = k2.copy()
            
            # Compute intermediate states
            state1 = state + self.dt * (a11 * k1 + a12 * k2)
            state2 = state + self.dt * (a21 * k1 + a22 * k2)
            
            # Evaluate derivatives at intermediate states
            k1 = self.derivatives(state1, False)
            k2 = self.derivatives(state2, False)
            
            # Check convergence
            error1 = np.max(np.abs(k1 - k1_old))
            error2 = np.max(np.abs(k2 - k2_old))
            
            if error1 < tolerance and error2 < tolerance:
                break
        
        # Update state using the computed k1 and k2
        new_state = state + self.dt * (b1 * k1 + b2 * k2)
        new_positions = new_state[0]
        new_velocities = new_state[1]
        
        return np.array([new_positions, new_velocities])
    
def main(segments, rope_weight, K, length_of_rope, mass_of_climber, climber_position, time, dt, damping, moisture, air_resistance=0, live_animation=False):

    anchor = np.zeros(2)

    rope = ImplicitRope(
        segments, rope_weight, 9.81, K,
        length_of_rope, mass_of_climber,
        climber_position, anchor,
        dt, time, damping, moisture, air_resistance
    )

    if live_animation:
        rope.run_with_live_animation(update_interval=50)
    else:
        rope.run()

    t = rope.timesteps
    x = np.linspace(0, t * rope.dt, t + 1)

    y = np.array(rope.p_hist)[:, -1, 1]

    plt.plot(x, y)
    plt.axhline(0, color="r", linestyle="--")
    plt.show()

    rope.plot_kinetic_energy()
    rope.rope_force()
    
if __name__ == "__main__":
    main(50, 5, 30000, 10, 75, np.array([0, 10]), 7, 0.0001, 800, 0, 0.5, True)
        