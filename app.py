import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# --- PHYSICS ENGINE ---
class NBodySolver:
    def __init__(self):
        self.G = 1.0  

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
                    if dist < 1e-5: dist = 1e-5 
                    dv_dt[i] += self.G * masses[j] * diff / (dist**3)
        
        return np.concatenate((dr_dt.flatten(), dv_dt.flatten()))

@st.cache_data
def run_physics_simulation(masses, initial_state, duration, steps):
    solver = NBodySolver()
    sol = solve_ivp(
        fun=lambda t, y: solver.equations(t, y, masses),
        t_span=(0, duration),
        y0=initial_state,
        method='DOP853',
        t_eval=np.linspace(0, duration, steps),
        rtol=1e-10, atol=1e-10
    )
    return sol.t, sol.y

# --- PRESETS ---
PRESETS = {
    "Figure-8 (Chenciner)": {
        't': 15, 'masses': [1, 1, 1],
        'pos': [[0.97000436,-0.24308753,0], [-0.97000436,0.24308753,0], [0,0,0]],
        'vel': [[0.46620368,0.43236573,0], [0.46620368,0.43236573,0], [-2*0.46620368,-2*0.43236573,0]]
    },
    "Broucke R7 (The Braid)": {
        't': 25, 'masses': [1, 1, 1],
        'pos': [[0.61856,-0.06316,0], [-0.61856,0.06316,0], [0,0,0]],
        'vel': [[-0.10399,0.59762,0], [-0.10399,0.59762,0], [2*0.10399,-2*0.59762,0]]
    },
    "Lagrange Equilateral": {
        't': 20, 'masses': [10, 0.1, 0.01],
        'pos': [[0,0,0], [2,0,0], [1, 1.732, 0]],
        'vel': [[0,0,0], [0,2.2,0], [-1.9, 1.1, 0]]
    },
    "Moth I (Li & Liao)": {
        't': 30, 'masses': [1, 1, 1],
        'pos': [[1,0,0], [-1,0,0], [0,0,0]],
        'vel': [[0.347111, 0.532728, 0], [0.347111, 0.532728, 0], [-2*0.347111, -2*0.532728, 0]]
    },
    "Yin-Yang (Li & Liao)": {
        't': 35, 'masses': [1, 1, 1],
        'pos': [[1,0,0], [-1,0,0], [0,0,0]],
        'vel': [[0.282699, 0.327209, 0], [0.282699, 0.327209, 0], [-2*0.282699, -2*0.327209, 0]]
    },
    "Dragonfly": {
        't': 40, 'masses': [1, 1, 1],
        'pos': [[1,0,0], [-1,0,0], [0,0,0]],
        'vel': [[0.080584, 0.588836, 0], [0.080584, 0.588836, 0], [-2*0.080584, -2*0.588836, 0]]
    },
    "Sitnikov (3D Vertical)": {
        't': 40, 'masses': [1, 1, 0.001],
        'pos': [[1,0,0], [-1,0,0], [0,0,1.5]],
        'vel': [[0, 0.5, 0], [0, -0.5, 0], [0, 0, 0]]
    },
    "Random Chaos": {
        't': 30, 'masses': [1.5, 1.0, 0.8],
        'pos': [[1.1,0.5,0], [-1.2,-0.3,0.1], [0.1,-0.1,-0.5]],
        'vel': [[-0.2,-0.1,0], [0.2,0.1,0.1], [0.05,0.05,0.2]]
    }
}

# --- WEB UI ---
st.set_page_config(page_title="3-Body Orbit Explorer", layout="wide")
st.title("🪐 3-Body Orbit Explorer")

if "run_sim" not in st.session_state:
    st.session_state.run_sim = False

# Sidebar Setup
st.sidebar.header("Simulation Controls")
selected_preset = st.sidebar.selectbox("Select Orbit Preset", list(PRESETS.keys()))
preset = PRESETS[selected_preset]

# User Settings
duration = st.sidebar.number_input("Duration", value=float(preset['t']), step=1.0)
steps = st.sidebar.number_input("Physics Data Points", value=3000, step=100)

# Visual Settings
st.sidebar.subheader("Visual Settings")
trail_mode = st.sidebar.selectbox("Path Mode", ["Static Path", "Growing Trail", "No Trail"])

# Note for users explaining the 3D rendering reality
st.sidebar.caption("⚠️ *Note: Due to 3D rendering constraints, camera controls are locked while the animation is playing. Pause the animation to rotate or zoom.*")

speed_options = {"0.25x (Slow)": {"dur": 120, "skip": 1}, 
                 "0.5x": {"dur": 60, "skip": 1}, 
                 "1x (Normal)": {"dur": 30, "skip": 1}, 
                 "2x": {"dur": 15, "skip": 1}, 
                 "4x (Fast)": {"dur": 15, "skip": 2}}
selected_speed = st.sidebar.select_slider("Playback Speed", options=list(speed_options.keys()), value="1x (Normal)")

st.sidebar.subheader("Initial Conditions")
masses = []
state_vector = []
colors = ['#FF4444', '#44FF44', '#4444FF']

for i in range(3):
    with st.sidebar.expander(f"Body {i+1}", expanded=False):
        m = st.number_input(f"Mass {i+1}", value=float(preset['masses'][i]), format="%.4f")
        masses.append(m)
        cols = st.columns(3)
        px = cols[0].number_input(f"Px {i+1}", value=float(preset['pos'][i][0]), format="%.4f")
        py = cols[1].number_input(f"Py {i+1}", value=float(preset['pos'][i][1]), format="%.4f")
        pz = cols[2].number_input(f"Pz {i+1}", value=float(preset['pos'][i][2]), format="%.4f")
        vx = cols[0].number_input(f"Vx {i+1}", value=float(preset['vel'][i][0]), format="%.4f")
        vy = cols[1].number_input(f"Vy {i+1}", value=float(preset['vel'][i][1]), format="%.4f")
        vz = cols[2].number_input(f"Vz {i+1}", value=float(preset['vel'][i][2]), format="%.4f")
        state_vector.extend([px, py, pz, vx, vy, vz])

if st.sidebar.button("🚀 Run Simulation", use_container_width=True):
    st.session_state.run_sim = True

# Execute if running
if st.session_state.run_sim:
    with st.spinner("Calculating physics and generating animation frames..."):
        t, y = run_physics_simulation(np.array(masses), np.array(state_vector), duration, int(steps))
        
        fig = go.Figure()
        
        # 1. Setup Base Traces based on Trail Mode
        if trail_mode != "No Trail":
            for i in range(3):
                if trail_mode == "Static Path":
                    fig.add_trace(go.Scatter3d(
                        x=y[i*3, :], y=y[i*3+1, :], z=y[i*3+2, :],
                        mode='lines', line=dict(color=colors[i], width=2),
                        opacity=0.3, name=f'Body {i+1} Path', hoverinfo='none'
                    ))
                elif trail_mode == "Growing Trail":
                    fig.add_trace(go.Scatter3d(
                        x=[y[i*3, 0]], y=[y[i*3+1, 0]], z=[y[i*3+2, 0]],
                        mode='lines', line=dict(color=colors[i], width=3),
                        opacity=0.8, name=f'Body {i+1} Trail', hoverinfo='none'
                    ))
                    
        # Add the starting markers for all modes
        for i in range(3):
            fig.add_trace(go.Scatter3d(
                x=[y[i*3, 0]], y=[y[i*3+1, 0]], z=[y[i*3+2, 0]],
                mode='markers', marker=dict(color=colors[i], size=10, line=dict(color='white', width=2)),
                name=f'Body {i+1}'
            ))

        # 2. Frame Logic & Speed Control
        frame_duration = speed_options[selected_speed]["dur"]
        skip_multiplier = speed_options[selected_speed]["skip"]
        base_skip = max(1, int(steps / 200)) 
        final_skip = base_skip * skip_multiplier
        
        frames = []
        for k in range(0, len(t), final_skip):
            frame_data = []
            
            if trail_mode == "Growing Trail":
                for i in range(3):
                    frame_data.append(go.Scatter3d(x=y[i*3, 0:k+1], y=y[i*3+1, 0:k+1], z=y[i*3+2, 0:k+1]))
                for i in range(3):
                    frame_data.append(go.Scatter3d(x=[y[i*3, k]], y=[y[i*3+1, k]], z=[y[i*3+2, k]]))
                frames.append(go.Frame(data=frame_data, name=str(k), traces=[0, 1, 2, 3, 4, 5]))
                
            elif trail_mode == "Static Path":
                for i in range(3):
                    frame_data.append(go.Scatter3d(x=[y[i*3, k]], y=[y[i*3+1, k]], z=[y[i*3+2, k]]))
                frames.append(go.Frame(data=frame_data, name=str(k), traces=[3, 4, 5]))
                
            elif trail_mode == "No Trail":
                for i in range(3):
                    frame_data.append(go.Scatter3d(x=[y[i*3, k]], y=[y[i*3+1, k]], z=[y[i*3+2, k]]))
                frames.append(go.Frame(data=frame_data, name=str(k), traces=[0, 1, 2]))

        fig.frames = frames

        # Play/Pause Buttons (redraw forced to True)
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.0, x=0.0, xanchor="left", yanchor="top", pad=dict(t=0, r=10),
                buttons=[
                    dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='black'
            ),
            paper_bgcolor='black', font=dict(color='white'),
            margin=dict(l=0, r=0, b=0, t=0), height=700, showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 Click **Run Simulation** in the sidebar to generate the animation!")