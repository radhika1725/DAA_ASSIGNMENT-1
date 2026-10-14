import streamlit as st
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque

# ============================================================================
# 1. DIJKSTRA'S ALGORITHM
# ============================================================================
def dijkstra(grid, start, goal, grid_size):
    if start == goal:
        return [start], [start], True
    
    distances = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
    distances[start] = 0
    pq = [(0, start)]
    visited, explored, came_from = set(), [], {}
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current in visited:
            continue
            
        visited.add(current)
        explored.append(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored, True
            
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    new_dist = current_dist + 1
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        came_from[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
                        
    return [], explored, False

# ============================================================================
# 2. A* ALGORITHM
# ============================================================================
def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def a_star(grid, start, goal, grid_size):
    if start == goal:
        return [start], [start], True
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from, explored, open_set_hash = {}, [], {start}
    
    g_score = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
    g_score[start] = 0
    f_score = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
    f_score[start] = heuristic(start, goal)
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.discard(current)
        explored.append(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored, True
            
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
                        
    return [], explored, False

# ============================================================================
# 3. BFS ALGORITHM (Breadth-First Search)
# ============================================================================
def bfs(grid, start, goal, grid_size):
    if start == goal:
        return [start], [start], True
    
    queue = deque([start])
    visited = {start}
    came_from, explored = {}, []
    
    while queue:
        current = queue.popleft()
        explored.append(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored, True
            
        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
                    
    return [], explored, False

# ============================================================================
# 4. JUMP POINT SEARCH (JPS) ALGORITHM 
# ============================================================================
def jump_point_search(grid, start, goal, grid_size):
    if start == goal:
        return [start], [start], True
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
    g_score[start] = 0
    f_score = {(i, j): float('inf') for i in range(grid_size) for j in range(grid_size)}
    f_score[start] = heuristic(start, goal)
    open_set_hash = {start}
    explored = []
    closed_set = set()
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current in closed_set:
            continue
        if current in open_set_hash:
            open_set_hash.discard(current)
            
        closed_set.add(current)
        explored.append(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], explored, True
        
        x, y = current
        
        # Cardinal directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                if neighbor not in closed_set:
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
        # Diagonal directions
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                if grid[x + dx][y] == 0 or grid[x][y + dy] == 0:
                    neighbor = (nx, ny)
                    if neighbor not in closed_set:
                        tentative_g = g_score[current] + 1.414 
                        if tentative_g < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                            if neighbor not in open_set_hash:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                                open_set_hash.add(neighbor)
    
    return [], explored, False

# ============================================================================
# 5. BIDIRECTIONAL SEARCH ALGORITHM
# ============================================================================
def construct_bidirectional_path(meet_point, fwd_came_from, bwd_came_from, start, goal):
    fwd_path = []
    curr = meet_point
    while curr in fwd_came_from:
        fwd_path.append(curr)
        curr = fwd_came_from[curr]
    fwd_path.append(start)
    fwd_path.reverse()
    
    bwd_path = []
    curr = meet_point
    while curr in bwd_came_from:
        curr = bwd_came_from[curr]
        bwd_path.append(curr)
        
    return fwd_path + bwd_path

def bidirectional_search(grid, start, goal, grid_size):
    if start == goal:
        return [start], [start], True
    
    fwd_q = deque([start])
    bwd_q = deque([goal])
    fwd_visited = {start}
    bwd_visited = {goal}
    fwd_came_from = {}
    bwd_came_from = {}
    explored = []
    
    while fwd_q and bwd_q:
        if fwd_q:
            curr_fwd = fwd_q.popleft()
            explored.append(curr_fwd)
            x, y = curr_fwd
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                    nxt = (nx, ny)
                    if nxt not in fwd_visited:
                        fwd_visited.add(nxt)
                        fwd_came_from[nxt] = curr_fwd
                        fwd_q.append(nxt)
                        if nxt in bwd_visited:
                            explored.append(nxt)
                            path = construct_bidirectional_path(nxt, fwd_came_from, bwd_came_from, start, goal)
                            return path, explored, True
        
        if bwd_q:
            curr_bwd = bwd_q.popleft()
            explored.append(curr_bwd)
            x, y = curr_bwd
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and grid[nx][ny] == 0:
                    nxt = (nx, ny)
                    if nxt not in bwd_visited:
                        bwd_visited.add(nxt)
                        bwd_came_from[nxt] = curr_bwd
                        bwd_q.append(nxt)
                        if nxt in fwd_visited:
                            explored.append(nxt)
                            path = construct_bidirectional_path(nxt, fwd_came_from, bwd_came_from, start, goal)
                            return path, explored, True
                        
    return [], explored, False


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Pathfinding Algorithm Web App on Interactive Grid Map", layout="wide")
st.title("🗺️ Pathfinding Algorithm Web App on Interactive Grid Map")
st.write("**Compare 5 famous pathfinding algorithms on an interactive 10×10 grid**")

# Define layout containers first (This determines visual order on screen)
grid_container = st.container()
controls_container = st.container()
info_container = st.container()

# Sidebar controls
st.sidebar.header("🎮 Controls")
grid_size = 10

if 'grid' not in st.session_state:
    st.session_state.grid = np.zeros((grid_size, grid_size), dtype=int)
    st.session_state.start = None
    st.session_state.goal = None
    st.session_state.path = []
    st.session_state.explored = []
    st.session_state.success = False
    st.session_state.algorithm = "A*"

def clear_results():
    st.session_state.path = []
    st.session_state.explored = []
    st.session_state.success = False

# Sidebar Selection
algo_options = [
    "A* (Smart - Uses Heuristic)", 
    "Dijkstra (Thorough - Checks All)",
    "BFS (Breadth-First - Flood Fill)",
    "JPS (Jump Point Search - Ultra Fast)",
    "Bidirectional (Meets in the Middle)"
]
selected_algo = st.sidebar.selectbox("Choose Algorithm:", algo_options, index=0)

if "A*" in selected_algo: st.session_state.algorithm = "A*"
elif "Dijkstra" in selected_algo: st.session_state.algorithm = "Dijkstra"
elif "BFS" in selected_algo: st.session_state.algorithm = "BFS"
elif "JPS" in selected_algo: st.session_state.algorithm = "JPS"
elif "Bidirectional" in selected_algo: st.session_state.algorithm = "Bidirectional"

mode = st.sidebar.radio(
    "**Select Mode:**",
    ["🟢 Set Start", "🔴 Set Goal", "⬛ Draw Obstacles", "🚀 Find Path", "🔄 Reset"]
)

st.sidebar.info(f"📌 Current Algorithm: **{st.session_state.algorithm}**")

# ============================================================================
# PROCESS CONTROLS FIRST 
# ============================================================================
with controls_container:
    st.divider()
    st.subheader("🎯 Instructions & Controls")

    if mode == "🟢 Set Start":
        st.write("**Place the starting position (green marker)**")
        col1, col2 = st.columns(2)
        with col1: x = st.slider("Column (X coordinate)", 0, grid_size - 1, 0, key="start_x")
        with col2: y = st.slider("Row (Y coordinate)", 0, grid_size - 1, 0, key="start_y")
        
        if st.button("✅ Confirm Start Position"):
            if st.session_state.grid[y][x] == 1:
                st.error("❌ Cannot place start on obstacle!")
            elif (y, x) == st.session_state.goal:
                st.error("❌ Start and goal cannot be the same!")
            else:
                st.session_state.start = (y, x)
                clear_results()

    elif mode == "🔴 Set Goal":
        st.write("**Place the goal position (red marker)**")
        col1, col2 = st.columns(2)
        with col1: x = st.slider("Column (X coordinate)", 0, grid_size - 1, 9, key="goal_x")
        with col2: y = st.slider("Row (Y coordinate)", 0, grid_size - 1, 9, key="goal_y")
        
        if st.button("✅ Confirm Goal Position"):
            if st.session_state.grid[y][x] == 1:
                st.error("❌ Cannot place goal on obstacle!")
            elif (y, x) == st.session_state.start:
                st.error("❌ Start and goal cannot be the same!")
            else:
                st.session_state.goal = (y, x)
                clear_results()

    elif mode == "⬛ Draw Obstacles":
        st.write("**Add or clear obstacles (black cells)**")
        col1, col2 = st.columns(2)
        with col1: x = st.slider("Column (X coordinate)", 0, grid_size - 1, 5, key="obs_x")
        with col2: y = st.slider("Row (Y coordinate)", 0, grid_size - 1, 5, key="obs_y")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚧 Add Obstacle"):
                if (y, x) == st.session_state.start or (y, x) == st.session_state.goal:
                    st.error("❌ Cannot place on start or goal!")
                else:
                    st.session_state.grid[y][x] = 1
                    clear_results()
        with col_btn2:
            if st.button("🟡 Clear Cell"):
                st.session_state.grid[y][x] = 0
                clear_results()

    elif mode == "🚀 Find Path":
        st.write("**Run the selected algorithm to find the shortest path**")
        if not st.session_state.start or not st.session_state.goal:
            st.error("❌ Set both start and goal first!")
        else:
            if st.button("🚀 Run Algorithm", key="run_algo"):
                args = (st.session_state.grid, st.session_state.start, st.session_state.goal, grid_size)
                
                if st.session_state.algorithm == "A*":
                    path, exp, succ = a_star(*args)
                elif st.session_state.algorithm == "Dijkstra":
                    path, exp, succ = dijkstra(*args)
                elif st.session_state.algorithm == "BFS":
                    path, exp, succ = bfs(*args)
                elif st.session_state.algorithm == "JPS":
                    path, exp, succ = jump_point_search(*args)
                elif st.session_state.algorithm == "Bidirectional":
                    path, exp, succ = bidirectional_search(*args)
                
                st.session_state.path = path
                st.session_state.explored = exp
                st.session_state.success = succ

    elif mode == "🔄 Reset":
        st.write("**Clear all obstacles, start, and goal**")
        if st.button("🔄 Clear Everything", key="reset_grid"):
            st.session_state.grid = np.zeros((grid_size, grid_size), dtype=int)
            st.session_state.start = None
            st.session_state.goal = None
            clear_results()

# ============================================================================
# RENDER GRID & STATUS
# ============================================================================
with grid_container:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Interactive Grid (10×10)")
        fig, ax = plt.subplots(figsize=(6, 6))
        
        display_grid = st.session_state.grid.copy().astype(int)
        
        if st.session_state.explored:
            for cell in st.session_state.explored:
                if display_grid[cell] == 0:
                    display_grid[cell] = 2  

        if st.session_state.path:
            for cell in st.session_state.path[1:-1]:
                if display_grid[cell] in [0, 2]:
                    display_grid[cell] = 3  

        if st.session_state.start:
            display_grid[st.session_state.start] = 4  
        if st.session_state.goal:
            display_grid[st.session_state.goal] = 5   

        colors = ['white', 'black', 'lightblue', 'yellow', 'green', 'red']
        cmap = ListedColormap(colors)
        
        im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=5)
        
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticklabels(np.arange(0, grid_size, 1))
        ax.set_yticklabels(np.arange(0, grid_size, 1))
        ax.tick_params(which='major', labelsize=8)
        
        ax.set_title("Pathfinding Grid (10×10) - X (Horizontal) × Y (Vertical)", fontsize=12, fontweight='bold')
        ax.set_xlabel("X Coordinate (Column)", fontsize=10)
        ax.set_ylabel("Y Coordinate (Row)", fontsize=10)
        
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("📊 Status Panel")
        
        algo_info = {
            "A*": ("🧠 A* Algorithm", "Smart + Fast. Guarantees shortest path."),
            "Dijkstra": ("🔍 Dijkstra Algorithm", "Thorough. Guarantees shortest path."),
            "BFS": ("🌊 BFS", "Flood-fill. Guarantees shortest path."),
            "JPS": ("⚡ Jump Point Search", "Ultra-fast. Skips many nodes. Guarantees shortest path."),
            "Bidirectional": ("🤝 Bidirectional", "Searches from both ends. Very fast!")
        }
        
        algo_title, algo_desc = algo_info.get(st.session_state.algorithm, ("Algorithm", ""))
        
        if st.session_state.algorithm == "A*":
            st.success(algo_title)
        elif st.session_state.algorithm == "Dijkstra" or st.session_state.algorithm == "BFS":
            st.info(algo_title)
        else:
            st.success(algo_title)
        
        st.caption(algo_desc)
        st.divider()
        
        if st.session_state.start:
            y, x = st.session_state.start
            st.success(f"✅ Start: Row {y}, Col {x}")
        else:
            st.info("❌ Start not set")
        
        if st.session_state.goal:
            y, x = st.session_state.goal
            st.success(f"✅ Goal: Row {y}, Col {x}")
        else:
            st.info("❌ Goal not set")
        
        st.write(f"🚧 Obstacles: {np.sum(st.session_state.grid == 1)}")
        st.divider()
        
        if st.session_state.success:
            st.success("✅ Path Found!")
            st.metric("Path Length", len(st.session_state.path))
            st.metric("Cells Explored", len(st.session_state.explored))
        elif st.session_state.explored and not st.session_state.success:
            st.error("❌ No path found!")
            st.write(f"Explored: {len(st.session_state.explored)} cells")

# ============================================================================
# ALGORITHM COMPARISON & INFO 
# ============================================================================
with info_container:
    st.divider()
    st.subheader("📚 Algorithm Comparison Guide")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["A*", "Dijkstra", "BFS", "JPS", "Bidirectional"])

    with tab1:
        st.write("### 🧠 A* Algorithm")
        st.markdown("- **Speed:** ⚡⚡⚡ Fastest\n- **Guarantee:** ✅ Shortest path guaranteed\n- **How it works:** Combines actual distance traveled + estimated distance to goal (heuristic)\n- **Best for:** General pathfinding, games, navigation")

    with tab2:
        st.write("### 🔍 Dijkstra's Algorithm")
        st.markdown("- **Speed:** ⚡⚡ Medium (slower than A*, faster than BFS on most maps)\n- **Guarantee:** ✅ Shortest path guaranteed\n- **How it works:** Exp outward evenly in all directions\n- **Best for:** Weighted graphs, network routing")

    with tab3:
        st.write("### 🌊 Breadth-First Search (BFS)")
        st.markdown("- **Speed:** ⚡⚡ Medium\n- **Guarantee:** ✅ Shortest path guaranteed (unweighted grids)\n- **How it works:** Flood-fill pattern, checks 1 step away, then 2, then 3...\n- **Best for:** Unweighted mazes, simple grids")

    with tab4:
        st.write("### ⚡ Jump Point Search (JPS)")
        st.markdown("- **Speed:** ⚡⚡⚡⚡ **Ultra-fast** (Fastest on uniform grids)\n- **Guarantee:** ✅ Shortest path guaranteed\n- **How it works:** Intelligently skips nodes by jumping over straight lines of walkable space\n- **Advantage:** Can be 10x faster than A* on large grids with few obstacles")

    with tab5:
        st.write("### ⚡ Bidirectional Search")
        st.markdown("- **Speed:** ⚡⚡⚡ Very fast (searches from both ends)\n- **Guarantee:** ✅ Shortest path guaranteed\n- **How it works:** Two simultaneous BFS searches meet in the middle\n- **Advantage:** Roughly 2x faster than standard algorithms")

    st.divider()
    st.subheader("🎨 Color Legend")
    legend_cols = st.columns(5)
    legend_cols[0].markdown("🟢 **Start** (Green)")
    legend_cols[1].markdown("🔴 **Goal** (Red)")
    legend_cols[2].markdown("⬛ **Obstacle** (Black)")
    legend_cols[3].markdown("🔵 **Explored** (Light Blue)")
    legend_cols[4].markdown("🟡 **Final Path** (Yellow)")

    st.divider()
    st.caption("92400118662 (V.Radhika)")