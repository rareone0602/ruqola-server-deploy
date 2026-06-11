"""Unit tests for ownership-aware pick_gpus / free_gpus / gpu_unavailability."""

ALICE = "alice"
BOB = "bob"


def _four_free():
    return [{"index": i, "memory_free_mb": 81920, "memory_total_mb": 81920,
             "utilization": 0} for i in range(4)]


# --- free-GPU picking (unchanged behavior, now with a user arg) --------------

def test_picks_first_two_when_idle(userspace_module):
    gpus = [
        {"index": 0, "memory_free_mb": 81920, "utilization": 0},
        {"index": 1, "memory_free_mb": 81920, "utilization": 0},
    ]
    assert userspace_module.pick_gpus(2, 70, gpus, [], ALICE) == [0, 1]


def test_returns_none_when_short(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 0}]
    assert userspace_module.pick_gpus(2, 70, gpus, [], ALICE) is None


def test_skips_busy_gpu(userspace_module):
    gpus = [
        {"index": 0, "memory_free_mb": 81920, "utilization": 0},
        {"index": 1, "memory_free_mb": 81920, "utilization": 0},
    ]
    running = [{"gpus": [0], "user": BOB}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE) == [1]


def test_blocks_on_high_util(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    assert userspace_module.pick_gpus(1, 70, gpus, [], ALICE) is None


def test_blocks_on_insufficient_memory(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 1024, "utilization": 0}]
    assert userspace_module.pick_gpus(1, 70, gpus, [], ALICE) is None


def test_exact_memory_fit(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 70 * 1024, "utilization": 0}]
    assert userspace_module.pick_gpus(1, 70, gpus, [], ALICE) == [0]


def test_default_pick_is_random_among_free(userspace_module, monkeypatch):
    """The default picker samples at random from ALL free GPUs (not lowest)."""
    seen = {}

    def fake_sample(pop, k):
        seen["pop"], seen["k"] = list(pop), k
        return sorted(pop)[-k:]            # deterministic stand-in: take the top k

    monkeypatch.setattr(userspace_module.random, "sample", fake_sample)
    got = userspace_module.pick_gpus(1, 70, _four_free(), [], ALICE)
    assert seen["pop"] == [0, 1, 2, 3] and seen["k"] == 1   # chose among all 4 free
    assert got == [3]                                       # per our stand-in


def test_random_result_is_subset_of_free(userspace_module):
    running = [{"gpus": [1], "user": BOB}]
    got = userspace_module.pick_gpus(2, 70, _four_free(), running, ALICE)
    assert len(got) == 2 and set(got) <= {0, 2, 3}          # never the busy GPU 1


def test_devices_pin_ok(userspace_module):
    assert userspace_module.pick_gpus(2, 70, _four_free(), [], ALICE,
                                      devices=[3, 1]) == [1, 3]


def test_devices_pin_rejected_when_held(userspace_module):
    running = [{"gpus": [1], "id": 5, "user": BOB}]
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      devices=[1]) is None


def test_devices_pin_rejected_when_busy_or_full(userspace_module):
    gpus = _four_free()
    gpus[2]["utilization"] = 95            # GPU 2 busy
    gpus[3]["memory_free_mb"] = 1024       # GPU 3 nearly full
    assert userspace_module.pick_gpus(1, 70, gpus, [], ALICE, devices=[2]) is None
    assert userspace_module.pick_gpus(1, 70, gpus, [], ALICE, devices=[3]) is None


def test_gpu_unavailability_reasons(userspace_module):
    gpus = _four_free()
    gpus[1]["utilization"] = 95
    gpus[2]["memory_free_mb"] = 1024
    running = [{"gpus": [0], "id": 7, "user": BOB}]
    r = userspace_module.gpu_unavailability([0, 1, 2, 9], gpus, running, 70, ALICE)
    blob = "\n".join(r)
    assert "held by gpuq job 7 (bob)" in blob          # GPU 0
    assert "GPU 1" in blob and "util" in blob           # GPU 1 busy
    assert "GPU 2" in blob and "free" in blob           # GPU 2 too full
    assert "GPU 9" in blob and "no such" in blob        # GPU 9 absent
    # a fully-free GPU produces no reason
    assert userspace_module.gpu_unavailability([3], gpus, running, 70, ALICE) == []


# --- ownership: "you own your allocated GPU" --------------------------------

def test_self_owned_gpu_is_selectable(userspace_module):
    """A GPU you already own is selectable even at high util (gate relaxed)."""
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) == [0]


def test_other_owned_gpu_blocked(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, BOB, devices=[0]) is None


def test_owned_gpu_vram_floor_enforced(userspace_module):
    """Owned cards still need GPU_OWN_MIN_FREE_GB of headroom."""
    floor_mb = userspace_module.GPU_OWN_MIN_FREE_GB * 1024
    gpus = [{"index": 0, "memory_free_mb": floor_mb - 1, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) is None
    assert userspace_module.free_gpus(70, gpus, running, ALICE) == []


def test_owned_gpu_skips_full_want_memory(userspace_module):
    """An owned card passes with only the small floor free, not the full want."""
    gpus = [{"index": 0, "memory_free_mb": 5 * 1024, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) == [0]


def test_cotenant_card_other_wins(userspace_module):
    """A card co-tenanted by another user is never offered to you."""
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1},
               {"gpus": [0], "user": BOB, "id": 9}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) is None
    r = userspace_module.gpu_unavailability([0], gpus, running, 70, ALICE)
    assert "held by gpuq job 9 (bob)" in "\n".join(r)


def test_gpu_unavailability_self_owned_ok(userspace_module):
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.gpu_unavailability([0], gpus, running, 70, ALICE) == []
    assert "held by" in "\n".join(
        userspace_module.gpu_unavailability([0], gpus, running, 70, BOB))


def test_missing_user_treated_as_foreign(userspace_module):
    """A running job with no recorded user is treated as someone else's."""
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 0}]
    running = [{"gpus": [0]}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) is None


def test_default_pick_prefers_free_over_owned(userspace_module):
    """With a free card available, the default picker uses it before stacking."""
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95},
            {"index": 1, "memory_free_mb": 81920, "utilization": 0}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE) == [1]


def test_default_pick_stacks_on_owned_when_no_free(userspace_module):
    """With no free card, the default picker stacks onto one you own."""
    gpus = [{"index": 0, "memory_free_mb": 81920, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE) == [0]
