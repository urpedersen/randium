def test_interaction_matrix():
    from randium import get_interactions
    M = 5
    I = get_interactions(M)

    # Confirm that I is an MxM symmetric matrix
    assert I.shape == (M, M), f'I.shape = {I.shape}'
    for u in range(M):
        for v in range(M):
            assert I[u, v] == I[v, u], f'I[{u},{v}] != I[{v},{u}]'
