"""Source-stock and usable-receipt overlap, never inferred from a pending task."""

import re

import composition_common01 as c


def assess(journal, transport, request_id, *, returned_at, deadline):
    pre = c.preflight()
    view = pre.inspect_cut(journal["cut"], consume_at=returned_at, deadline=deadline)
    cut, phases = journal["cut"], journal["phases"]
    digest = pre.frozen.module("prefill_lab.contracts").digest
    c.require(phases["cut_canonical_sha256"] == digest(cut)
              and phases["source_at"] == cut["cut_at"]
              and phases["available_at"] == cut["validated_at"]
              and phases["classified_at"] <= returned_at, "inflight_source_receipt_binding")
    domain = cut["common_clock_proof"]["clock_domain"]
    c.require(transport["request_id"] == request_id
              and transport["client_clock_domain"] == domain == phases["clock_domain"]
              and transport["dispatch_attempted"] is True, "inflight_request_or_clock_binding")
    sent, source, available, received = map(pre.number, (
        transport["client_send_at"], cut["cut_at"], returned_at, transport["client_receive_at"]))
    c.require(sent <= received and source <= available <= deadline, "inflight_future_or_unknown_receipt")
    c.require(sent < source, "inflight_source_not_after_dispatch")
    stock = view["full_stock"]
    running, waiting = set(stock["scheduler_state"]["running"]), set(stock["scheduler_state"]["waiting"])
    queued = {item["request_id"] for route in stock["queue_oldest_first"] for item in route["work"]}
    ghosts = set(stock["queued_ghost_request_ids"])
    candidates = [nid for nid in cut["native_observation"]["native_phase"]["publications"]
                  if re.fullmatch("chatcmpl-" + re.escape(request_id) + "-[0-9a-f]{8}", nid)]
    c.require(len(candidates) <= 1, "duplicate_managed_native_identity")
    native_id = candidates[0] if candidates else None
    lifecycle = cut["native_observation"]["native_phase"]["lifecycle"].get(native_id)
    target_stock = bool(native_id and native_id in running | waiting | queued)
    native_kind = ("running" if native_id in running else "waiting" if native_id in waiting else
                   "issued_ghost" if native_id in ghosts else "issued" if native_id in queued else
                   "publication_only" if native_id and lifecycle is None else
                   "native_finished_no_issued_stock" if native_id else "absent")
    source_overlap = sent < source < received
    usable_overlap = source_overlap and source <= available < received
    status = ("nonempty_available_while_pending" if target_stock and usable_overlap else
              "source_after_outcome" if not source_overlap else
              "available_after_outcome" if not usable_overlap else
              "publication_only" if native_kind == "publication_only" else "empty_or_unattributed_stock")
    return dict(schema="team16-source-bound-inflight-evidence-v1", status=status,
        request_id=request_id, native_id=native_id, native_kind=native_kind,
        target_scheduler_or_issued_stock=target_stock,
        source_within_transport=source_overlap, available_while_transport_pending=usable_overlap,
        nonempty_overlap=target_stock and usable_overlap, source_at=source, available_at=available,
        client_send_at=sent, client_receive_at=received, clock_domain=domain,
        stock_counts=dict(running=len(running), waiting=len(waiting), issued_routes=len(stock["queue_oldest_first"]),
                          issued_request_ids=len(queued), ghosts=len(ghosts),
                          zero_work_routes=sum(not row["work"] for row in stock["queue_oldest_first"]),
                          frontend_unfinished=stock["base"]["frontend_unfinished"],
                          pending_batches=stock["base"]["pending_batches"]),
        cut_sha256=digest(cut), actual_application_at=None, acquisition_token=None, request_token=None,
        deployable=False, qualified_model=False, prediction_qualified=False, native_mpc_permitted=False,
        scope="One source-bound target inventory with its collector receipt before the managed terminal receipt; not persistent state or exact action time.")
