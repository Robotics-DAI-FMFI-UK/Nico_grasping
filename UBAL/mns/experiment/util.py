def humanreadible_runtime(runtime):
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}'.format(int(h), int(m), round(s))