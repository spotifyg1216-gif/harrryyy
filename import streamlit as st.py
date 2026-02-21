import streamlit as st
import pandas as pd
import numpy as np
from geomdl import fitting, operations
import pyvista as pv
from stpyvista import stpyvista

# --- INBUILT CMM ALGORITHM LOGIC ---
def compute_cmm_logic(points, u_size, v_size, density):
    """
    Core logic to convert raw points into a NURBS surface 
    and generate CMM approach vectors (I, J, K).
    """
    degree = 3
    # Fit the NURBS surface to the car headlight reflector data
    surf = fitting.interpolate_surface(points, u_size, v_size, degree, degree)
    
    u_vals = np.linspace(0, 1, density)
    v_vals = np.linspace(0, 1, density)
    
    path_results, plot_points, vectors = [], [], []

    for u in u_vals:
        for v in v_vals:
            # Get 3D Point (X, Y, Z)
            p = operations.evaluate_surface(surf, [[u, v]])[0]
            
            # Calculate Surface Normal for Probe Approach (I, J, K)
            derivs = operations.evaluate_derivatives(surf, u, v, order=1)
            norm = np.cross(derivs[1][0], derivs[0][1])
            unit_norm = norm / np.linalg.norm(norm)
            
            path_results.append({
                'X': p[0], 'Y': p[1], 'Z': p[2], 
                'I': unit_norm[0], 'J': unit_norm[1], 'K': unit_norm[2]
            })
            plot_points.append(p)
            vectors.append(unit_norm)
            
    return pd.DataFrame(path_results), np.array(plot_points), np.array(vectors)

# --- APP INTERFACE ---
st.set_page_config(page_title="CMM Reflector Design", layout="wide")
st.title("🗜️ CMM Freeform Path Generator")
st.markdown("Automated Path Planning for **Car Headlight Reflectors**.")

st.sidebar.header("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Point Cloud (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    pts = df[['x', 'y', 'z']].values.tolist()
    
    u_grid = st.sidebar.number_input("NURBS Grid U", value=10)
    v_grid = st.sidebar.number_input("NURBS Grid V", value=10)
    sample_res = st.sidebar.slider("Sampling Density", 5, 40, 15)

    if st.sidebar.button("Generate Inspection Path"):
        res_df, coords, norms = compute_cmm_logic(pts, u_grid, v_grid, sample_res)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("3D Probe Path Preview")
            plotter = pv.Plotter(window_size=[600, 600])
            plotter.background_color = "white"
            plotter.add_mesh(pv.PolyData(coords), color="red", point_size=10, render_points_as_spheres=True)
            plotter.add_arrows(coords, norms, mag=5, color="blue")
            stpyvista(plotter)
            
        with col2:
            st.subheader("CMM Path Table (DMIS Compatible)")
            st.dataframe(res_df)
            st.download_button("Download CSV for CMM", res_df.to_csv(index=False), "cmm_path.csv")
else:
    st.info("Upload a CSV file with x, y, z coordinates to start the algorithm.")