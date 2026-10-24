"""Serialization of voxel records and evaluation outputs."""
import csv,json
LABEL_COLORS = {0: (0, 0, 0), 1: (70, 130, 180), 2: (34, 139, 34), 3: (128, 128, 128), 4: (220, 20, 60), 5: (255, 215, 0), 6: (160, 82, 45), 7: (255, 255, 255)}
def label_to_color(label):
    return LABEL_COLORS.get(
        int(label),
        (
            int((label * 53) % 255),
            int((label * 97) % 255),
            int((label * 193) % 255),
        ),
    )


def save_records_json(path, records):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def save_metrics_csv(path, records):
    fields = [
        "key_x",
        "key_y",
        "key_z",
        "center_x",
        "center_y",
        "center_z",
        "dominant_class",
        "op",
        "num_observed_frames",
        "vss",
        "sbd",
        "entropy",
        "entropy_norm",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in records:
            key = r["key"]
            center = r["center"]

            writer.writerow(
                {
                    "key_x": key[0],
                    "key_y": key[1],
                    "key_z": key[2],
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "dominant_class": r["dominant_class"],
                    "op": r["op"],
                    "num_observed_frames": r["num_observed_frames"],
                    "vss": r["vss"],
                    "sbd": r["sbd"],
                    "entropy": r["entropy"],
                    "entropy_norm": r["entropy_norm"],
                }
            )


def save_voxel_ply(path, records):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(records)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("property float vss\n")
        f.write("property float sbd\n")
        f.write("property float entropy_norm\n")
        f.write("property int op\n")
        f.write("end_header\n")

        for r in records:
            x, y, z = r["center"]
            label = int(r["dominant_class"])
            red, green, blue = label_to_color(label)

            vss = -1.0 if r["vss"] is None else float(r["vss"])
            sbd = -1.0 if r["sbd"] is None else float(r["sbd"])
            ent = -1.0 if r["entropy_norm"] is None else float(r["entropy_norm"])
            op = int(r["op"])

            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{red} {green} {blue} "
                f"{label} {vss:.6f} {sbd:.6f} {ent:.6f} {op}\n"
            )

