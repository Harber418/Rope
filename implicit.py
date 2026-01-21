from rope import Rope
from rope import main
import numpy as np
from matplotlib import pyplot as plt

class ImplicitRope(Rope):

    def method(self, u_n):
        """method for updating the state of the system.
           u_n: {arr} current state of the system. u_n[0] contains
           all mass position. u_n[1] contains all mass velocities"""
        """2-stage Gauss-Legendre method"""
        # 2-stage Gauss-Legendre Butcher tableau coefficients
        a11 = 0.25
        a12 = 0.25 - np.sqrt(3) / 6.0
        a21 = 0.25 + np.sqrt(3) / 6.0
        a22 = 0.25
        b1 = 0.5
        b2 = 0.5
        
        # Initial guess for k1 and k2 using explicit Euler
        U1 = self.derivatives(u_n, True)
        U2 = self.derivatives(u_n, False)
        
        # Fixed-point iteration to solve the implicit system
        max_iterations = 10
        tolerance = 1e-10
        
        for iteration in range(max_iterations):
            U1_old = U1.copy()
            U2_old = U2.copy()
            
            # Compute intermediate states
            state1 = u_n + self.dt * (a11 * U1 + a12 * U2)
            state2 = u_n + self.dt * (a21 * U1 + a22 * U2)
            
            # Evaluate derivatives at intermediate states
            U1 = self.derivatives(state1, False)
            U2 = self.derivatives(state2, False)
            
            # Check convergence
            error1 = np.max(np.abs(U1 - U1_old))
            error2 = np.max(np.abs(U2 - U2_old))
            
            if error1 < tolerance and error2 < tolerance:
                break
        
        # return u_n+1 using the computed U1 and U2
        return u_n + self.dt * (b1 * U1 + b2 * U2)
    
def main(segments, rope_weight, K, length_of_rope, mass_of_climber, climber_position, time, dt, damping, moisture, air_resistance=0):

    anchor = np.zeros(2)

    rope = ImplicitRope(
        segments, rope_weight, 9.81, K,
        length_of_rope, mass_of_climber,
        climber_position, anchor,
        dt, time, damping, moisture, air_resistance
    )

    rope.run_with_live_animation(update_interval=50)

    t = rope.timesteps
    x = np.linspace(0, t * rope.dt, t + 1)

    y = np.array(rope.p_hist)[:, -1, 1]

    plt.plot(x, y)
    plt.axhline(0, color="r", linestyle="--")
    plt.show()

    rope.plot_kinetic_energy()
    rope.rope_force()
    rope.save_history("implicit_rope_simulation.npz")
    
if __name__ == "__main__":
    main(30, 5, 40000, 10, 75, np.array([5, 0]), 10, 0.001, 30, 0, 0.3)
        