import numpy as np
import matplotlib.pyplot as plt


def position(sets):
    r1,r2,r3,r4,r5 = sets
    data_r1 = np.load(r1)
    data_r2 = np.load(r2)
    data_r3 = np.load(r3)
    data_r4 = np.load(r4)
    data_r5 = np.load(r5)
    p_hist_r1 = data_r1['p_hist']
    p_hist_r2 = data_r2['p_hist']
    p_hist_r3 = data_r3['p_hist']
    p_hist_r4 = data_r4['p_hist']
    p_hist_r5 = data_r5['p_hist']
    
    fall_factor_rk4 = p_hist_r1['fall_factor']
    time = data_r1['time']
    t = p_hist_r1.shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt

    y = np.array(p_hist_r1)[:, -1, 1]
    y2 = np.array(p_hist_r2)[:, -1, 1]
    y3 = np.array(p_hist_r3)[:, -1, 1]
    y4 = np.array(p_hist_r4)[:, -1, 1]
    y5 = np.array(p_hist_r5)[:, -1, 1]
    
    plt.plot(x, y, label='year 1 ')
    plt.plot(x, y2, label='year 2')
    plt.plot(x, y3, label='year 3')
    plt.plot(x, y4, label='year 4')
    plt.plot(x, y5, label='year 5')
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel('Time (s)')
    plt.ylabel('Climber Y Position (m)')
    plt.title(f'Climber Y Position vs Time (units: m, s, kg), fall factor {fall_factor_rk4}')
    plt.legend()
    plt.show()
    

def plot_force_comparison(sets):
    """Load force histories from two files and plot their average rope force on the same graph."""
    r1,r2,r3,r4,r5 = sets
    data_r1 = np.load(r1)
    data_r2 = np.load(r2)
    data_r3 = np.load(r3)
    data_r4 = np.load(r4)
    data_r5 = np.load(r5)

    
    f_hist_rk1 = data_r1['f_hist']
    f_hist_rk2 = data_r2['f_hist']
    f_hist_rk3 = data_r3['f_hist']
    f_hist_rk4 = data_r4['f_hist']
    f_hist_rk5 = data_r5['f_hist']
    data = [f_hist_rk1,f_hist_rk2,f_hist_rk3,f_hist_rk4,f_hist_rk5]
    fall_factor_rk4 = data_r1['fall_factor']

    plotting = []
    for i in data:
        total_forces = []
        for forces in i:
            # no anchor no climber 
            rope_forces = forces[1:-1]
            # axis=1 is used as forces are in 2D
            norms = np.linalg.norm(rope_forces, axis=1)
            
            total_force = np.sum(norms)
            total_forces.append(total_force)

        plotting.append(total_forces)


    t_rk4 = np.arange(len(plotting[0]))

    time = data_r1['time']
    t = np.array(plotting[0]).shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt

    plt.figure(figsize=(10, 6))
    plt.plot(x, plotting[0], label='1 year', alpha = 0.8, c = "r")
    plt.plot(x, plotting[1], label='2 yeaer', alpha = 0.8, c = "b")
    plt.plot(x, plotting[2], label='3 yeaer', alpha = 0.8, c = "b")
    plt.plot(x, plotting[3], label='4 yeaer', alpha = 0.8, c = "b")
    plt.plot(x, plotting[4], label='5 yeaer', alpha = 0.8, c = "b")

    plt.xlabel('Time (s)')
    plt.ylabel('Rope Force (kN)')
    plt.title(f'Rope Force Comparison : fall factor {fall_factor_rk4}')
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
    time = data_rk4['time']
    t = ke_rk4.shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt
    plt.figure(figsize=(10, 6))
    plt.plot(x, ke_rk4, label='RK4 Method')
    plt.plot(x, ke_impl, label='Implicit Method')
    plt.xlabel('Time (s)')
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

    time = data_rk4['time']
    t = f_hist_rk4.shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt
    plt.figure(figsize=(10, 6))
    plt.plot(x, tensions_rk4, label='RK4 Method', alpha = 0.8, c = "r")
    plt.plot(x, tensions_impl, label='Implicit Method', alpha = 0.8, c = "b")
    plt.xlabel('Time (s)')
    plt.ylabel('Maximum Tension (N)')
    plt.title(f'Maximum Tension Comparison: fall factor {fall_factor_rk4}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def climber_jerk(rk4_file, implicit_file):
    data_rk4 = np.load(rk4_file)
    data_impl = np.load(implicit_file)
    f_hist_rk4 = data_rk4['f_hist']
    f_hist_impl = data_impl['f_hist']
    fall_factor_rk4 = data_rk4['fall_factor']
    masses_rk4 = data_rk4['masses']
    masses_impl = data_impl['masses'] 
    time = data_rk4['time']
    t = f_hist_rk4.shape[0]

    dt = time / (t - 1)
    x = np.arange(t) * dt  
    x_jerk = x[101:]

    a_hist = []
    for f in f_hist_impl:
        a = f[-1] / masses_impl[-1]
        a_hist.append(a)

    a_hist = np.array(a_hist)
    jerk = np.diff(a_hist, axis=0) / dt
    jerk_mag = np.linalg.norm(jerk, axis=1)

    a_hist_rk4 = []
    for f in f_hist_rk4:
        a = f[-1] / masses_rk4[-1]
        a_hist_rk4.append(a)

    a_hist_rk4 = np.array(a_hist)
    jerk_rk4 = np.diff(a_hist, axis=0) / dt
    jerk_mag_rk4 = np.linalg.norm(jerk, axis=1)

    plt.figure()
    plt.plot(x_jerk,jerk_mag[100:],label='implicit')
    plt.plot(x_jerk,jerk_mag_rk4[100:], label='RK4')
    plt.title(f'Jerk experiensed by the climber, fall factor {fall_factor_rk4}')
    plt.xlabel('time (s)')
    plt.ylabel('magnitude of jerk')
    plt.legend()
    plt.show()
    


def main():
    rope1 = 'implicit_rope_simulation1.npz'
    rope2 = 'implicit_rope_simulation2.npz'
    rope3 = 'implicit_rope_simulation3.npz'
    rope4 = 'implicit_rope_simulation4.npz'
    rope5 = 'implicit_rope_simulation5.npz'

    sets = [rope1,rope2,rope3,rope4,rope5]
    position(sets)
    plot_force_comparison(sets)

    #plot_kinetic_energy_comparison(sets)
    #tension(sets)
    #climber_jerk(sets)


if __name__ == "__main__":
    main()