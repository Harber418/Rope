import numpy as np
import matplotlib.pyplot as plt


def position(rk4_file, impl_file):
    data_rk4 = np.load(rk4_file)
    data_impl = np.load(impl_file)
    p_hist_rk4 = data_rk4['p_hist']
    p_hist_impl = data_impl['p_hist']
    fall_factor_rk4 = data_rk4['fall_factor']
    time = data_rk4['time']
    t = p_hist_rk4.shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt

    y = np.array(p_hist_rk4)[:, -1, 1]
    y_impl = np.array(p_hist_impl)[:, -1, 1]
    plt.plot(x, y, label='RK4')
    plt.plot(x, y_impl, label='Implicit')
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel('Time (s)')
    plt.ylabel('Climber Y Position (m)')
    plt.title(f'Climber Y Position vs Time (units: m, s, kg), fall factor {fall_factor_rk4}')
    plt.legend()
    plt.show()
    

def plot_force_comparison(rk4_file, implicit_file):
    """Load force histories from two files and plot their average rope force on the same graph."""
    data_rk4 = np.load(rk4_file)
    data_impl = np.load(implicit_file)
    f_hist_rk4 = data_rk4['f_hist']
    f_hist_impl = data_impl['f_hist']
    fall_factor_rk4 = data_rk4['fall_factor']
    fall_factor_impl = data_impl['fall_factor']

    total_forces_rk4 = []
    for forces in f_hist_rk4:
        # no anchor no climber 
        rope_forces = forces[1:-1]
        # axis=1 is used as forces are in 2D
        norms = np.linalg.norm(rope_forces, axis=1)
        
        total_force = np.sum(norms)
        total_forces_rk4.append(total_force)
    total_forces_imp = []
    for forces in f_hist_impl:
        # no anchor no climber 
        rope_forces = forces[1:-1]
        # axis=1 is used as forces are in 2D
        norms = np.linalg.norm(rope_forces, axis=1)

        total_force = np.sum(norms)
        total_forces_imp.append(total_force)
    t_rk4 = np.arange(len(total_forces_rk4))
    t_impl = np.arange(len(total_forces_imp))

    plt.figure(figsize=(10, 6))
    plt.plot(t_rk4, total_forces_rk4, label='RK4 Method', alpha = 0.8, c = "r")
    plt.plot(t_impl, total_forces_imp, label='Implicit Method', alpha = 0.8, c = "b")
    plt.xlabel('Time Step')
    plt.ylabel('Rope Force (N)')
    plt.title(f'Rope Force Comparison : fall factor {fall_factor_rk4}, iterations {len(total_forces_rk4)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_kinetic_energy_comparison(rk4_file, implicit_file, masses_rk4, masses_impl):
    """
    Plot the total kinetic energy for both RK4 and implicit methods.
    - rk4_file, implicit_file: .npz files with v_hist
    - masses_rk4, masses_impl: 1D arrays of masses for each method (shape: [n_masses])
    """
    data_rk4 = np.load(rk4_file)
    data_impl = np.load(implicit_file)
    v_hist_rk4 = data_rk4['v_hist']
    v_hist_impl = data_impl['v_hist']
    fall_factor_rk4 = data_rk4['fall_factor']
    masses_rk4 = data_rk4['masses']
    masses_impl = data_impl['masses']


    def total_ke(v_hist, masses):
        ke_total = []
        for velocities in v_hist:
            v_squared = np.sum(velocities**2, axis=1)
            ke = 0.5 * masses * v_squared
            ke_total.append(np.sum(ke))
        return np.array(ke_total)

    ke_rk4 = total_ke(v_hist_rk4, masses_rk4)
    ke_impl = total_ke(v_hist_impl, masses_impl)
    t_rk4 = np.arange(len(ke_rk4))
    t_impl = np.arange(len(ke_impl))

    plt.figure(figsize=(10, 6))
    plt.plot(t_rk4, ke_rk4, label='RK4 Method')
    plt.plot(t_impl, ke_impl, label='Implicit Method')
    plt.xlabel('Time Step')
    plt.ylabel('Total Kinetic Energy (J)')
    plt.title(f'Total Kinetic Energy Comparison : fall factor {fall_factor_rk4}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def tension(rk4_file, implicit_file):
    data_rk4 = np.load(rk4_file)
    data_impl = np.load(implicit_file)
    f_hist_rk4 = data_rk4['f_hist']
    f_hist_impl = data_impl['f_hist']
    fall_factor_rk4 = data_rk4['fall_factor']
    tensions_rk4 = []
    for forces in f_hist_rk4:
        tensions_rk4.append(forces[-1])
    
    tensions_impl = []
    for forces in f_hist_impl:
        tensions_impl.append(forces[-1])

    t_rk4 = np.arange(len(tensions_rk4))
    t_impl = np.arange(len(tensions_impl))

    plt.figure(figsize=(10, 6))
    plt.plot(t_rk4, tensions_rk4, label='RK4 Method', alpha = 0.8, c = "r")
    plt.plot(t_impl, tensions_impl, label='Implicit Method', alpha = 0.8, c = "b")
    plt.xlabel('Time Step')
    plt.ylabel('Maximum Tension (N)')
    plt.title(f'Maximum Tension Comparison: fall factor {fall_factor_rk4}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    rk4_file = 'rope_simulation_data_rk4.npz'
    implicit_file = 'implicit_rope_simulation.npz'

    position(rk4_file, implicit_file)
    plot_force_comparison(rk4_file, implicit_file)

    data_rk4 = np.load(rk4_file)
    data_impl = np.load(implicit_file)
    masses_rk4 = data_rk4['masses']
    masses_impl = data_impl['masses']

    plot_kinetic_energy_comparison(rk4_file, implicit_file, masses_rk4, masses_impl)
    tension(rk4_file, implicit_file)


if __name__ == "__main__":
    main()