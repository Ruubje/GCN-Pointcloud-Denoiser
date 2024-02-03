from meshplot import plot as mp_plot
from numpy import array as np_array, vstack as np_vstack, zeros as np_zeros

def visualize(vs, es):
    vsizes = np_array(list(map(lambda x: x.shape[0], vs)))
    csvsizes = vsizes.cumsum()
    c = np_zeros(csvsizes[-1])
    c[csvsizes[:-1]] = 1
    c.cumsum()
    v = np_vstack(vs)

    plot = mp_plot(v, c=c, shading={"point_size": 0.005})

    if not es:
        return
    
    esizes = np_array(list(map(lambda x: x.shape[1], es)))
    csesizes = esizes.cumsum()
    a = np_zeros(csesizes[-1], dtype=int)
    a[csesizes[:-1]] = csvsizes[:-1]
    a.cumsum()
    e = np_vstack(es)
    e += a

    starts = v[e[0]]
    ends = v[e[1]]

    plot.add_lines(starts, ends)

def visualize_coloring(v, c, f=None):
    mode = f is None
    a = np_zeros(len(v) if mode else len(f), dtype=int)
    for i, ci in enumerate(c):
        a[ci] = i + 1
    return mp_plot(v, f=f, c=a)
