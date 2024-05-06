# degeneracy_paths

Data path:
home/karen.kang/LIGOSURF23/Q/paths/

2D/{mass}/{mass}_{idx}.pkl
mass = ['lowmass', ’90','270']
idx = 0-8
*270 is not usable, correlation does not recover in 2D

3D/{mass}/{mass}_{idx}.pkl
mass = ['lowmass', ’90','270']
idx = 0-8

5D/{mass}/{mass}_{idx}.pkl
mass = ['90vs', '270vs',’90','270']
idx = 0-9
for '90vs', '270vs', idx = vs0-9  e.g. 5D/270vs/270_vs0.pkl
*There is no 30 due to not enough effective samples in higher dimensions
*There is 90, 270 but I haven’t fixed the point spacings
*check 90vs and 270vs, this is where I tested points with same effective spins but varying inplane spins. 

If you are interested in running the mapping at reference injections of your interest, here are the settings I used. 
In general, you would want 500+ points after rejection sampling. 

<img width="678" alt="image" src="https://github.com/yixinkang/degeneracy_paths/assets/112017703/cd2dec40-cb7c-4b38-8f64-aabb9af8b557">
