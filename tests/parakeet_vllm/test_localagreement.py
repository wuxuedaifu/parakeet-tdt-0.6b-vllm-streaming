from parakeet_vllm.realtime.localagreement import LocalAgreement

def test_commits_only_agreed_prefix_n2():
    la = LocalAgreement(n=2)
    assert la.commit(["the", "cat"]) == []                 # first hop: nothing agreed yet
    assert la.commit(["the", "cat", "sat"]) == ["the", "cat"]  # "the cat" agreed across 2 hops
    assert la.commit(["the", "cat", "sat", "on"]) == ["sat"]   # "sat" now agreed

def test_never_retracts_on_tail_change():
    la = LocalAgreement(n=2)
    la.commit(["a", "b", "X"])
    la.commit(["a", "b", "Y"])          # commits "a","b"; tail differs
    out = la.commit(["a", "b", "Y", "z"])  # "Y" agreed now
    assert out == ["Y"]
    assert la.committed == ["a", "b", "Y"]

def test_committed_prefix_is_stable_once_emitted():
    la = LocalAgreement(n=2)
    la.commit(["one", "two"])
    la.commit(["one", "two", "three"])   # commits one,two
    # a later hypothesis that cleanly extends the committed prefix does not disturb it
    la.commit(["one", "two", "three", "four"])
    assert la.committed[:2] == ["one", "two"]

def test_divergent_longer_prefix_does_not_corrupt_committed():
    """Agreed prefix that is longer than committed but diverges early must not retract."""
    from parakeet_vllm.realtime.localagreement import LocalAgreement
    la = LocalAgreement(n=2)
    # Note: hyp1 uses "c" so common prefix with hyp2 is ["a","b"] only, not ["a","b","e"].
    assert la.commit(["a", "b", "c"]) == []            # 1 hyp, nothing
    assert la.commit(["a", "b", "e", "f"]) == ["a", "b"]  # commits a,b
    assert la.commit(["a", "c", "d"]) == []            # agreed=["a"], shorter → blocked
    # next window agrees on ["a","c","d"] (len 3 > committed len 2) but diverges
    # at position 1 ("c" != committed "b") → must NOT retract/corrupt committed
    assert la.commit(["a", "c", "d", "g"]) == []
    assert la.committed == ["a", "b"]


def test_empty_hypothesis():
    la = LocalAgreement(n=2)
    assert la.commit([]) == []
    assert la.committed == []
