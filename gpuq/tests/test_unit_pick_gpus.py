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
    """Even with a tiny -m, an owned card needs the GPU_OWN_MIN_FREE_GB floor of
    headroom — the floor binds when the requested -m is below it."""
    floor_mb = userspace_module.GPU_OWN_MIN_FREE_GB * 1024
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    below = [{"index": 0, "memory_free_mb": floor_mb - 1, "utilization": 95}]
    assert userspace_module.pick_gpus(1, 0, below, running, ALICE, devices=[0]) is None
    assert userspace_module.free_gpus(0, below, running, ALICE) == []
    at = [{"index": 0, "memory_free_mb": floor_mb, "utilization": 95}]
    assert userspace_module.pick_gpus(1, 0, at, running, ALICE, devices=[0]) == [0]


def test_owned_gpu_honors_want_memory(userspace_module):
    """Issue #1 (Bug 2): an owned card is gated by the requested free-VRAM filter,
    not just the small floor. A card with 5 GB free can't satisfy -m 70, so it is
    NOT selectable (pre-fix it stacked regardless); a smaller -m it CAN satisfy
    still stacks (util ignored on a card you own)."""
    gpus = [{"index": 0, "memory_free_mb": 5 * 1024, "utilization": 95}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, gpus, running, ALICE, devices=[0]) is None
    assert userspace_module.free_gpus(70, gpus, running, ALICE) == []
    assert userspace_module.pick_gpus(1, 4, gpus, running, ALICE, devices=[0]) == [0]


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


def test_default_pick_wont_stack_on_own_busy_card(userspace_module):
    """Issue #1 (Bug 1): a near-full owned card no longer auto-stacks under a -m it
    can't meet. With every card either held by another user or too full for the
    request, the default picker returns None so a --queue submit WAITS for a free
    card instead of stacking onto a busy one."""
    gpus = [{"index": 0, "memory_free_mb": 37 * 1024,    # mine, busy
             "memory_total_mb": 140 * 1024, "utilization": 100},
            {"index": 1, "memory_free_mb": 37 * 1024,    # someone else's, busy
             "memory_total_mb": 140 * 1024, "utilization": 100}]
    running = [{"gpus": [0], "user": ALICE, "id": 1},
               {"gpus": [1], "user": BOB, "id": 2}]
    assert userspace_module.pick_gpus(1, 130, gpus, running, ALICE) is None


# --- per-user concurrent-card hard cap (max_gpus_per_user_hard) --------------

def _three_alice_jobs():
    return [{"gpus": [0], "user": ALICE, "id": 1},
            {"gpus": [1], "user": ALICE, "id": 2},
            {"gpus": [2], "user": ALICE, "id": 3}]


def test_hard_cap_blocks_new_card(userspace_module):
    """Holding hard_cap distinct cards, a request needing a NEW card is refused
    even though a free card exists."""
    assert userspace_module.pick_gpus(4, 70, _four_free(), [], ALICE,
                                      hard_cap=3) is None
    running = _three_alice_jobs()
    # want 1 more distinct card while already holding 3 -> only stacking left;
    # with all owned cards unable to take the request... here they CAN stack,
    # so instead pin the free card to force the new-card path:
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      devices=[3], hard_cap=3) is None


def test_hard_cap_redirects_to_stacking(userspace_module):
    """At the cap, the default picker stacks onto an owned card instead of
    claiming a free one."""
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      hard_cap=1) == [0]


def test_hard_cap_partial_free_partial_stack(userspace_module):
    """Holding 2 with cap 3, a 2-GPU job takes exactly one new card and stacks
    the other onto a card already held."""
    running = [{"gpus": [0], "user": ALICE, "id": 1},
               {"gpus": [1], "user": ALICE, "id": 2}]
    got = userspace_module.pick_gpus(2, 70, _four_free(), running, ALICE,
                                     hard_cap=3)
    assert got is not None and len(got) == 2
    assert len(set(got) & {0, 1}) == 1 and len(set(got) & {2, 3}) == 1


def test_hard_cap_under_cap_prefers_free(userspace_module):
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    got = userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                     hard_cap=3)
    assert got is not None and got[0] in (1, 2, 3)


def test_hard_cap_devices_pin_owned_card_still_ok(userspace_module):
    """Pinning a card you already hold never breaches the cap (stacking)."""
    running = _three_alice_jobs()
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      devices=[0], hard_cap=3) == [0]


def test_hard_cap_counts_cotenant_cards(userspace_module):
    """A card you share with another user still counts toward YOUR held total
    (and is not stackable, since the other user holds it too)."""
    running = [{"gpus": [0], "user": ALICE, "id": 1},
               {"gpus": [0], "user": BOB, "id": 2}]
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      hard_cap=1) is None


def test_hard_cap_zero_disables(userspace_module):
    running = _three_alice_jobs()
    assert userspace_module.pick_gpus(1, 70, _four_free(), running, ALICE,
                                      hard_cap=0) == [3]


def test_hard_cap_over_cap_user_can_still_pin_owned_card(userspace_module):
    """Rollout day: a user already holding MORE cards than the cap (claimed
    before the cap existed) may still pin/stack onto a card they own — only
    NEW cards are charged against the headroom, exactly like the default
    picker's stacking redirect."""
    running = [{"gpus": [i], "user": ALICE, "id": i} for i in range(4)]
    assert userspace_module.pick_gpus(1, 2, _four_free(), running, ALICE,
                                      devices=[2], hard_cap=3) == [2]
    assert userspace_module.pick_gpus(1, 2, _four_free(), running, ALICE,
                                      hard_cap=3) is not None


def test_gpu_unavailability_reports_card_cap(userspace_module):
    """The pinned-refusal reason text mirrors the picker's cap check: a pin
    blocked purely by the per-user card cap must say so, never return an
    empty reason list for a refused submit."""
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    r = userspace_module.gpu_unavailability([1], _four_free(), running, 70,
                                            ALICE, hard_cap=1)
    assert r and "card cap" in "\n".join(r)
    # pinning the card you already hold is stacking, not a cap violation
    assert userspace_module.gpu_unavailability([0], _four_free(), running, 70,
                                               ALICE, hard_cap=1) == []


def test_normal_waiter_can_claim_detects_wedged_waiter(userspace_module):
    """A normal waiter wedged by the card cap must not count as claimable:
    deprioritized submitters yield only to waiters who can actually take a
    slot, else a free card idles while the low-priority job starves."""
    gpus = [{"index": 0, "memory_free_mb": 1024, "utilization": 100},
            {"index": 1, "memory_free_mb": 81920, "utilization": 0}]
    running = [{"gpus": [0], "user": BOB, "id": 1,
                "host": userspace_module.HOST}]
    waiter = {"priority": "normal", "host": userspace_module.HOST,
              "user": BOB, "gpu_count": 1, "memory_gb": 70, "devices": None}
    can = userspace_module._normal_waiter_can_claim
    assert can([waiter], running, gpus, 1) is False   # capped + no stack VRAM
    assert can([waiter], running, gpus, 0) is True    # cap off: free GPU 1
    assert can([], running, gpus, 1) is False         # nobody waiting


def test_owned_gpu_unavailability_honors_memory(userspace_module):
    """Issue #1: the pinned-GPU reason text mirrors the picker — an owned busy card
    that can't meet -m is reported as too-full, not silently usable."""
    gpus = [{"index": 0, "memory_free_mb": 37 * 1024,
             "memory_total_mb": 140 * 1024, "utilization": 100}]
    running = [{"gpus": [0], "user": ALICE, "id": 1}]
    # -m it can't meet: reason given, and consistent with free_gpus == []
    r = userspace_module.gpu_unavailability([0], gpus, running, 130, ALICE)
    assert r and "GPU 0" in r[0] and "130 GB" in r[0]
    assert userspace_module.free_gpus(130, gpus, running, ALICE) == []
    # -m it can meet: no reason, and the picker would stack
    assert userspace_module.gpu_unavailability([0], gpus, running, 30, ALICE) == []
    assert userspace_module.free_gpus(30, gpus, running, ALICE) == [0]
