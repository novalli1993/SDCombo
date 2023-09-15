import os

data_path = "J:/Dataset/Stanford2D3D"
fold = {"training": [[1, 2, 3, 4, 6], [1, 2, 3, 4, 6], [2, 4, 5]],
        "validation": [[5], [2, 4], [1, 3, 6]]}
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth"]

check_area = [1]

for a in check_area:
    for t in types:
        image_dir = os.path.join(data_path, area[a - 1], t)
        assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
        for _, _, p in os.walk(image_dir):
            if t == "rgb":
                images = [os.path.join(image_dir, x) for x in p]
            elif t == "semantic":
                annotations = [os.path.join(image_dir, x) for x in p]
            elif t == "depth":
                depth = [os.path.join(image_dir, x) for x in p]
assert (len(images) == len(annotations))
assert (len(images) == len(depth))