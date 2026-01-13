from rope import Rope
from rope import main
import numpy as np
from matplotlib import pyplot as plt

class VerletRope(Rope):

    def method(self, state):
        """use the velocity verlet method to update the state"""
        positions, velocities = state
        
        # Get current acceleration (a(t))
        _, acceleration_t = self.derivatives(state, True)
        
        # Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_positions = positions + velocities * self.dt + 0.5 * acceleration_t * self.dt**2
        
        # Create temporary state with new positions and old velocities
        temp_state = np.array([new_positions, velocities])
        
        # Get new acceleration (a(t+dt))
        _, acceleration_t_plus_dt = self.derivatives(temp_state, False)
        
        # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        new_velocities = velocities + 0.5 * (acceleration_t + acceleration_t_plus_dt) * self.dt
        
        return np.array([new_positions, new_velocities])
    
def main(segments, rope_weight, K, length_of_rope, mass_of_climber, climber_position, time, dt, damping, moisture, air_resistance=0, live_animation=False):

    anchor = np.zeros(2)

    rope = VerletRope(
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
    
if __name__ == "__main__":
    main(50, 5, 30000, 10, 75, np.array([0, 5]), 30, 0.001, 100, 0, 0, True)

    # note: i see no visible improvement with verlet over rk4 in this scenario
        