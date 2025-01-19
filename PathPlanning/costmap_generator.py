import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import json
from pathlib import Path

def load_map(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=float)

def calc_cost_map(bin_map, max_c=100.0, decay=0.5):
    mask = bin_map == 1
    dists = distance_transform_edt(mask)
    cost_map = np.full_like(bin_map, np.nan, dtype=float)
    max_dist = np.max(dists[mask]) if np.any(mask) else 1
    norm_dists = dists / max_dist
    cost_map[mask] = max_c * (1 - np.exp(-decay * norm_dists[mask]))
    return cost_map

def plot_cost_map(cost_map, out_path):
    plt.figure(figsize=(12, 12))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('white')
    plt.imshow(cost_map, cmap=cmap, origin='lower')
    h, w = cost_map.shape
    plt.plot([0, w - 1], [0, 0], 'k-', linewidth=1)
    plt.plot([0, w - 1], [h - 1, h - 1], 'k-', linewidth=1)
    plt.plot([0, 0], [0, h - 1], 'k-', linewidth=1)
    plt.plot([w - 1, w - 1], [0, h - 1], 'k-', linewidth=1)
    plt.axis('off')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white', edgecolor='none')
    plt.close()

def save_cost_json(cost_map, out_path):
    cost_list = np.where(np.isnan(cost_map), 0, cost_map).tolist()
    with open(out_path, 'w') as f:
        json.dump(cost_list, f, indent=2)

def process_dir(in_dir, out_dir, max_c=100.0, decay=0.5):
    out_dir = Path(out_dir)
    json_dir = out_dir / "json"
    img_dir = out_dir / "img"
    
    for d in [json_dir, img_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    in_dir = Path(in_dir)
    json_files = list(in_dir.glob('*.json'))    
    for json_path in json_files:
        try:
            bin_map = load_map(json_path)
            cost_map = calc_cost_map(bin_map, max_c, decay)
            img_path = img_dir / f"{json_path.stem}_cost.png"
            json_out = json_dir / f"{json_path.stem}_cost.json"
            
            plot_cost_map(cost_map, img_path)
            save_cost_json(cost_map, json_out)
        except Exception as e:
            pass

if __name__ == "__main__":
    in_dir = "map_data"
    out_dir = "cost_maps"
    process_dir(
        in_dir,
        out_dir,
        max_c=100.0,
        decay=0.5
    )
