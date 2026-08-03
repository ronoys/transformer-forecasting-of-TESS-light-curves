# run this ONCE on the login node (has internet) to build tess_windows.npz
# usage: python build_data.py

import real_data_msml612_demo as m

m.make_tess_windows("tess_windows.npz")
print("done: tess_windows.npz built")
