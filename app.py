from flask import Flask, jsonify, request, send_from_directory
from cube import Cube
from model import ValueNetwork
import torch
import numpy as np
import heapq
import os

app    = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cube  = Cube()
model = None

def try_load_model():
    global model
    if os.path.exists('model.pth'):
        model = ValueNetwork().to(DEVICE)
        model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
        model.eval()
        print(f'Model loaded (device={DEVICE})')

try_load_model()


def solve_astar(cube_in, max_nodes=150000, w=3.0):
    """Weighted A*: priority = g + w*h. Higher w trusts the heuristic and dives to solved."""
    if cube_in.is_solved():
        return []

    start_key = cube_in.state.tobytes()
    sv = torch.tensor(cube_in.get_state_vector(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        start_h = -model(sv).max().item()

    heap    = [(w * start_h, 0, 0, start_key, [])]
    visited = set()
    node_id = 0
    temp    = Cube()

    while heap and node_id < max_nodes:
        _, g, _, state_key, moves = heapq.heappop(heap)

        if state_key in visited:
            continue
        visited.add(state_key)

        temp.state = np.frombuffer(state_key, dtype=np.int8).reshape(6, 2, 2).copy()

        nbr_vecs  = []
        nbr_keys  = []
        nbr_moves = []

        for move in range(18):
            saved = temp.state.copy()
            temp.apply_move(move)
            key = temp.state.tobytes()
            if key not in visited:
                if temp.is_solved():
                    return moves + [move]
                nbr_vecs.append(temp.get_state_vector())
                nbr_keys.append(key)
                nbr_moves.append(moves + [move])
            temp.state = saved

        if not nbr_vecs:
            continue

        t = torch.tensor(np.array(nbr_vecs), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            h_vals = -model(t).cpu().numpy().max(axis=1)

        for key, mv_list, h in zip(nbr_keys, nbr_moves, h_vals):
            node_id += 1
            g_new = len(mv_list)
            heapq.heappush(heap, (g_new + w * float(h), g_new, node_id, key, mv_list))

    return None


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/state')
def state():
    return jsonify({'state': cube.to_dict()})

@app.route('/scramble', methods=['POST'])
def scramble():
    n = request.json.get('n', 5)
    cube.__init__()
    moves = cube.scramble(n)
    # build per-move intermediate states so the frontend can animate each step
    temp = Cube()
    steps = []
    for m in moves:
        temp.apply_move(m)
        steps.append({'move_name': Cube.MOVE_NAMES[m], 'state': temp.to_dict()})
    return jsonify({
        'state': cube.to_dict(),
        'moves': [Cube.MOVE_NAMES[m] for m in moves],
        'steps': steps,
    })

@app.route('/reset', methods=['POST'])
def reset():
    cube.__init__()
    return jsonify({'state': cube.to_dict()})

@app.route('/move', methods=['POST'])
def move():
    idx = request.json.get('move')
    cube.apply_move(idx)
    return jsonify({'state': cube.to_dict(), 'solved': bool(cube.is_solved())})

@app.route('/solve', methods=['POST'])
def solve():
    if model is None:
        return jsonify({'error': 'Model not trained yet. Run train.py first.'}), 400

    solution_moves = solve_astar(cube, max_nodes=50000)

    if solution_moves is None:
        return jsonify({'solution': [], 'solved': False,
                        'error': 'AI could not solve this — model may still be training, or try a smaller scramble (3-5 moves)'})

    # replay the moves to get per-step states
    temp = cube.copy()
    steps = []
    for m in solution_moves:
        temp.apply_move(m)
        steps.append({'move_idx': m, 'move_name': Cube.MOVE_NAMES[m], 'state': temp.to_dict()})
        if temp.is_solved():
            break

    # apply to real cube
    for m in solution_moves:
        cube.apply_move(m)

    return jsonify({'solution': steps, 'solved': bool(cube.is_solved())})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
