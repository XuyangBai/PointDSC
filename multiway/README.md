## Multiway Registration Experiments 

The code is mainly following this excellent [tutorial.](http://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html) 

### Data
You first need to download the `Augmented_ICL` dataset from this [website](http://redwood-data.org/indoor/dataset.html) and make the file structure like this,
```
├── livingroom1-simulated
│   ├── color
│   │   └── *.png
│   └── depth
│       └── *.png
├── livingroom1-traj.txt
├── livingroom2-simulated
│   ├── color
│   └── depth
├── livingroom2-traj.txt
├── office1-simulated
│   ├── color
│   └── depth
├── office1-traj.txt
├── office2-simulated
│   ├── color
│   └── depth
└── office2-traj.txt

```

### Make Fragments

This step creates the point cloud fragments from RGB-D sequences.

```bash
python multiway/make_fragments.py redwood_simulated/livingroom1-simulated.json
```
The generated ply files will be saved in `livingroom-simulated/fragments/`.


### Compute FPFH/FCGF descriptors

Then we compute the local descriptors for the ply files to construct the putative correspondences as the input to our network

```bash
python misc/cal_fpfh.py
# python misc/cal_fcgf.py
```

To compute the fcgf descriptor, you need to first download the pre-trained weight on from [here](http://node2.chrischoy.org/data/projects/DGR/ResUNetBN2C-feat32-3dmatch-v0.05.pth).

### Multiway Registration

```bash
python multiway/test_multi_ate.py --chosen_snapshot [snapshot]
```

The combined point cloud will be saved and visualized then.
