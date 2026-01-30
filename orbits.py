import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
import threading

# --- PHYSICS ENGINE ---

class NBodySolver:
    def __init__(self):
        self.G = 1.0  # Normalized Gravitational Constant

    def equations(self, t, state, masses):
        n = len(masses)
        r = state[:3*n].reshape((n, 3))
        v = state[3*n:].reshape((n, 3))
        
        dr_dt = v
        dv_dt = np.zeros((n, 3))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = r[j] - r[i]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-5: dist = 1e-5 # Softening
                    dv_dt[i] += self.G * masses[j] * diff / (dist**3)
        
        return np.concatenate((dr_dt.flatten(), dv_dt.flatten()))

    def solve_chunked(self, masses, initial_state, t_span, steps, progress_callback):
        t_eval = np.linspace(t_span[0], t_span[1], steps)
        chunk_size = steps // 10
        current_state = initial_state
        current_time = t_span[0]
        
        full_results = {'t': [], 'y': []}
        dt_chunk = (t_span[1] - t_span[0]) / 10
        
        try:
            for i in range(10):
                t_end = current_time + dt_chunk
                
                sol = solve_ivp(
                    fun=lambda t, y: self.equations(t, y, masses),
                    t_span=(current_time, t_end),
                    y0=current_state,
                    method='DOP853',
                    t_eval=np.linspace(current_time, t_end, chunk_size),
                    rtol=1e-10, atol=1e-10
                )
                
                if i == 0:
                    full_results['t'].append(sol.t)
                    full_results['y'].append(sol.y)
                else:
                    full_results['t'].append(sol.t[1:])
                    full_results['y'].append(sol.y[:, 1:])
                
                current_state = sol.y[:, -1]
                current_time = t_end
                progress_callback((i + 1) * 10)
                
            final_t = np.concatenate(full_results['t'])
            final_y = np.concatenate(full_results['y'], axis=1)
            return final_t, final_y
            
        except Exception as e:
            raise e

# --- GUI APPLICATION ---

class ThreeBodyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3-Body Orbit Explorer")
        self.root.geometry("1400x950")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.solver = NBodySolver()
        self.simulation_data = None
        self.masses = []
        self.anim = None
        self.is_paused = False
        self.current_frame = 0  # Manual frame counter for speed control
        
        # Visualization Defaults
        self.trail_length_var = tk.StringVar(value="Long")
        self.planet_size_var = tk.DoubleVar(value=8.0)
        self.sim_speed_var = tk.DoubleVar(value=0.0) # 0 is neutral
        
        self.create_layout()
        self.setup_presets()
        self.load_preset("Figure-8 (Chenciner)") 

    def create_layout(self):
        left_panel = ttk.Frame(self.root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        right_panel = ttk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Presets
        ttk.Label(left_panel, text="Select Orbit", font=("Arial", 11, "bold")).pack(pady=5)
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(left_panel, textvariable=self.preset_var, state="readonly", width=30)
        self.preset_combo.pack(fill=tk.X, pady=5)
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self.load_preset(self.preset_var.get()))

        # 2. Input Fields
        inputs_frame = ttk.LabelFrame(left_panel, text="Initial Conditions", padding=5)
        inputs_frame.pack(fill=tk.X, pady=10)
        
        self.entries = []
        self.body_notebook = ttk.Notebook(inputs_frame)
        self.body_notebook.pack(fill=tk.X)
        
        colors = ['#FF4444', '#44FF44', '#4444FF'] 
        for i in range(3):
            page = ttk.Frame(self.body_notebook)
            self.body_notebook.add(page, text=f"Body {i+1}")
            fields = ['Mass', 'Pos X', 'Pos Y', 'Pos Z', 'Vel X', 'Vel Y', 'Vel Z']
            body_entries = {}
            for row, field in enumerate(fields):
                ttk.Label(page, text=field, foreground=colors[i]).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                ent = ttk.Entry(page, width=15)
                ent.grid(row=row, column=1, padx=5, pady=2)
                body_entries[field] = ent
            self.entries.append(body_entries)

        # 3. Physics Settings
        settings_frame = ttk.LabelFrame(left_panel, text="Simulation Settings", padding=5)
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Duration:").grid(row=0, column=0)
        self.duration_ent = ttk.Entry(settings_frame, width=8)
        self.duration_ent.insert(0, "20")
        self.duration_ent.grid(row=0, column=1)
        
        ttk.Label(settings_frame, text="Data Points:").grid(row=1, column=0)
        self.steps_ent = ttk.Entry(settings_frame, width=8)
        self.steps_ent.insert(0, "3000")
        self.steps_ent.grid(row=1, column=1)

        # 4. Visualization Settings
        vis_frame = ttk.LabelFrame(left_panel, text="Visualization", padding=5)
        vis_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(vis_frame, text="Trail Length:").pack(anchor="w")
        self.trail_combo = ttk.Combobox(vis_frame, textvariable=self.trail_length_var, state="readonly", 
                                        values=["None", "Short", "Long", "Infinite"])
        self.trail_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(vis_frame, text="Planet Size:").pack(anchor="w", pady=(5,0))
        self.size_scale = tk.Scale(vis_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                                   variable=self.planet_size_var)
        self.size_scale.pack(fill=tk.X)

        # 5. Calculation Controls
        self.solve_btn = ttk.Button(left_panel, text="RUN SIMULATION", command=self.start_calculation_thread)
        self.solve_btn.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.export_btn = ttk.Button(left_panel, text="Export CSV", command=self.export_csv, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=5)

        # 6. Telemetry
        telemetry_frame = ttk.LabelFrame(left_panel, text="Live Data", padding=5)
        telemetry_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.telemetry_label = tk.Label(telemetry_frame, text="Ready.", justify=tk.LEFT, font=("Courier", 8))
        self.telemetry_label.pack(anchor="nw")

        # --- Plot Area ---
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Dark Theme Setup
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('#0f0f0f')
        self.ax.set_axis_off() 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Animation Controls (Play/Pause/Speed)
        controls_frame = ttk.Frame(right_panel)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.pause_btn = ttk.Button(controls_frame, text="Pause/Play", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=10)
        
        # Speed Control
        ttk.Label(controls_frame, text="Playback Speed:").pack(side=tk.LEFT, padx=(20, 5))
        # Slider from -5 (Slow) to 0 (Normal) to +10 (Fast)
        self.speed_scale = tk.Scale(controls_frame, from_=-5, to=10, orient=tk.HORIZONTAL, showvalue=0, length=200, resolution=1)
        self.speed_scale.set(0) # Default Middle (Normal Speed)
        self.speed_scale.pack(side=tk.LEFT)
        ttk.Label(controls_frame, text="(Slow <--- Normal ---> Fast)").pack(side=tk.LEFT, padx=5)

    def setup_presets(self):
        self.presets = {}
        # 1. Figure-8
        self.presets["Figure-8 (Chenciner)"] = {
            't': 15,
            'bodies': [
                {'m':1, 'p':[0.97000436,-0.24308753,0], 'v':[0.46620368,0.43236573,0]},
                {'m':1, 'p':[-0.97000436,0.24308753,0], 'v':[0.46620368,0.43236573,0]},
                {'m':1, 'p':[0,0,0], 'v':[-2*0.46620368,-2*0.43236573,0]}
            ]
        }
        # 2. Broucke R7
        self.presets["Broucke R7 (The Braid)"] = {
            't': 25,
            'bodies': [
                {'m':1, 'p':[0.61856,-0.06316,0], 'v':[-0.10399,0.59762,0]},
                {'m':1, 'p':[-0.61856,0.06316,0], 'v':[-0.10399,0.59762,0]},
                {'m':1, 'p':[0,0,0], 'v':[2*0.10399,-2*0.59762,0]}
            ]
        }
        # 3. Lagrange Equilateral
        self.presets["Lagrange Equilateral"] = {
            't': 20,
            'bodies': [
                {'m':10, 'p':[0,0,0], 'v':[0,0,0]},
                {'m':0.1, 'p':[2,0,0], 'v':[0,2.2,0]},
                {'m':0.01, 'p':[1, 1.732, 0], 'v':[-1.9, 1.1, 0]}
            ]
        }
        # 4. Moth I
        self.presets["Moth I (Li & Liao)"] = {
            't': 30,
            'bodies': [
                {'m':1, 'p':[1,0,0], 'v':[0.347111, 0.532728, 0]},
                {'m':1, 'p':[-1,0,0], 'v':[0.347111, 0.532728, 0]},
                {'m':1, 'p':[0,0,0], 'v':[-2*0.347111, -2*0.532728, 0]}
            ]
        }
        # 5. Yin-Yang
        self.presets["Yin-Yang (Li & Liao)"] = {
            't': 35,
            'bodies': [
                {'m':1, 'p':[1,0,0], 'v':[0.282699, 0.327209, 0]},
                {'m':1, 'p':[-1,0,0], 'v':[0.282699, 0.327209, 0]},
                {'m':1, 'p':[0,0,0], 'v':[-2*0.282699, -2*0.327209, 0]}
            ]
        }
        # 6. Dragonfly
        self.presets["Dragonfly"] = {
            't': 40,
            'bodies': [
                {'m':1, 'p':[1,0,0], 'v':[0.080584, 0.588836, 0]},
                {'m':1, 'p':[-1,0,0], 'v':[0.080584, 0.588836, 0]},
                {'m':1, 'p':[0,0,0], 'v':[-2*0.080584, -2*0.588836, 0]}
            ]
        }
        # 7. Sitnikov
        self.presets["Sitnikov (3D Vertical)"] = {
            't': 40,
            'bodies': [
                {'m':1, 'p':[1,0,0], 'v':[0, 0.5, 0]},
                {'m':1, 'p':[-1,0,0], 'v':[0, -0.5, 0]},
                {'m':0.001, 'p':[0,0,1.5], 'v':[0, 0, 0]}
            ]
        }
        # 8. Random
        self.presets["Random Chaos"] = {
            't': 30,
            'bodies': [
                {'m':1.5, 'p':[1.1,0.5,0], 'v':[-0.2,-0.1,0]},
                {'m':1.0, 'p':[-1.2,-0.3,0.1], 'v':[0.2,0.1,0.1]},
                {'m':0.8, 'p':[0.1,-0.1,-0.5], 'v':[0.05,0.05,0.2]}
            ]
        }

        self.preset_combo['values'] = list(self.presets.keys())
        self.preset_combo.current(0)

    def load_preset(self, name):
        data = self.presets[name]
        self.duration_ent.delete(0, tk.END)
        self.duration_ent.insert(0, str(data['t']))
        
        for i, body in enumerate(data['bodies']):
            entries = self.entries[i]
            entries['Mass'].delete(0, tk.END); entries['Mass'].insert(0, str(body['m']))
            
            p = body['p']
            entries['Pos X'].delete(0, tk.END); entries['Pos X'].insert(0, str(p[0]))
            entries['Pos Y'].delete(0, tk.END); entries['Pos Y'].insert(0, str(p[1]))
            entries['Pos Z'].delete(0, tk.END); entries['Pos Z'].insert(0, str(p[2]))
            
            v = body['v']
            entries['Vel X'].delete(0, tk.END); entries['Vel X'].insert(0, str(v[0]))
            entries['Vel Y'].delete(0, tk.END); entries['Vel Y'].insert(0, str(v[1]))
            entries['Vel Z'].delete(0, tk.END); entries['Vel Z'].insert(0, str(v[2]))

    def get_inputs(self):
        masses = []
        state_vector = []
        try:
            for i in range(3):
                masses.append(float(self.entries[i]['Mass'].get()))
            for i in range(3):
                state_vector.extend([float(self.entries[i]['Pos X'].get()), float(self.entries[i]['Pos Y'].get()), float(self.entries[i]['Pos Z'].get())])
            for i in range(3):
                state_vector.extend([float(self.entries[i]['Vel X'].get()), float(self.entries[i]['Vel Y'].get()), float(self.entries[i]['Vel Z'].get())])
            t_span = (0, float(self.duration_ent.get()))
            steps = int(self.steps_ent.get())
            return np.array(masses), np.array(state_vector), t_span, steps
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers.")
            return None, None, None, None

    def start_calculation_thread(self):
        masses, state, t_span, steps = self.get_inputs()
        if masses is None: return
        
        self.solve_btn.config(state=tk.DISABLED, text="CALCULATING...")
        self.progress_var.set(0)
        self.telemetry_label.config(text="Solving differential equations...")
        
        thread = threading.Thread(target=self.run_calculation, args=(masses, state, t_span, steps))
        thread.start()

    def run_calculation(self, masses, state, t_span, steps):
        try:
            t, y = self.solver.solve_chunked(
                masses, state, t_span, steps, 
                progress_callback=lambda p: self.root.after(0, lambda: self.progress_var.set(p))
            )
            self.masses = masses
            self.simulation_data = {'t': t, 'y': y}
            self.root.after(0, self.start_animation)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.solve_btn.config(state=tk.NORMAL, text="RUN SIMULATION"))

    def start_animation(self):
        if self.anim: self.anim.event_source.stop()
        
        t = self.simulation_data['t']
        y = self.simulation_data['y']
        
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.set_facecolor('black')
        
        # Calculate bounds
        all_x = y[0:9:3, :].flatten()
        all_y = y[1:9:3, :].flatten()
        all_z = y[2:9:3, :].flatten()
        max_range = np.max([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]) / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Create Visuals
        self.lines = [self.ax.plot([], [], [], '-', lw=1.5, alpha=0.8)[0] for _ in range(3)]
        self.points = [self.ax.plot([], [], [], 'o')[0] for _ in range(3)]
        
        colors = ['#FF4444', '#44FF44', '#4444FF'] 
        for i, (l, p) in enumerate(zip(self.lines, self.points)):
            l.set_color(colors[i])
            p.set_color(colors[i])
            p.set_markeredgecolor('white')

        self.current_frame = 0 # Reset frame

        def update(_):
            if self.is_paused: return
            
            # --- SPEED CONTROL LOGIC ---
            speed_val = self.speed_scale.get()
            
            # 1. Base jump (1 frame)
            step_jump = 1 
            delay = 30 # standard 30ms delay
            
            if speed_val > 0:
                # Fast forward: Increase jump, keep delay low
                # e.g., Speed 5 -> Jump 6 frames at a time
                step_jump = 1 + int(speed_val) 
                delay = 20
            elif speed_val < 0:
                # Slow motion: Jump 1, but increase delay
                # e.g., Speed -5 -> Delay becomes 30 + 5*30 = 180ms
                step_jump = 1
                delay = 30 + (abs(speed_val) * 40)

            # Apply speed settings
            if self.anim: 
                self.anim.event_source.interval = delay

            # Advance Frame
            self.current_frame += step_jump
            
            # Loop handling
            if self.current_frame >= len(t):
                self.current_frame = 0
            
            idx = self.current_frame
            
            # --- DRAWING ---
            
            # Telemetry
            txt = f"Time: {t[idx]:.2f} / {t[-1]:.2f}\n"
            for i in range(3):
                vx = y[9 + i*3, idx]
                vy = y[9 + i*3+1, idx]
                vz = y[9 + i*3+2, idx]
                vel = np.sqrt(vx**2 + vy**2 + vz**2)
                txt += f"Body {i+1}: Vel={vel:.3f}\n"
            self.telemetry_label.config(text=txt)

            # Trail Logic
            trail_type = self.trail_length_var.get()
            if trail_type == "None":
                start = idx
            elif trail_type == "Short":
                start = max(0, idx - 50)
            elif trail_type == "Long":
                start = max(0, idx - 200)
            else: # Infinite
                start = 0

            # Size Logic
            current_size = self.planet_size_var.get()
            
            for i in range(3):
                xi = y[i*3, start:idx+1]
                yi = y[i*3+1, start:idx+1]
                zi = y[i*3+2, start:idx+1]
                
                self.lines[i].set_data(xi, yi)
                self.lines[i].set_3d_properties(zi)
                
                self.points[i].set_data([y[i*3, idx]], [y[i*3+1, idx]])
                self.points[i].set_3d_properties([y[i*3+2, idx]])
                self.points[i].set_markersize(current_size)

        # Important: frames=None allows infinite looping controlled by our manual index
        self.anim = FuncAnimation(self.fig, update, frames=None, interval=30, blit=False)
        self.canvas.draw()
        self.export_btn.config(state=tk.NORMAL)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="Resume" if self.is_paused else "Pause")

    def export_csv(self):
        if self.simulation_data is None: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            t = self.simulation_data['t']
            y = self.simulation_data['y']
            df = pd.DataFrame({'Time': t})
            for i in range(3):
                df[f'B{i+1}_x'] = y[i*3, :]
                df[f'B{i+1}_y'] = y[i*3+1, :]
                df[f'B{i+1}_z'] = y[i*3+2, :]
                df[f'B{i+1}_vx'] = y[9+i*3, :]
                df[f'B{i+1}_vy'] = y[9+i*3+1, :]
                df[f'B{i+1}_vz'] = y[9+i*3+2, :]
            df.to_csv(path, index=False)
            messagebox.showinfo("Export", "Saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThreeBodyApp(root)
    root.mainloop()