# Research context and transfer limits

These public primary sources were retrieved during the audit. They motivate
questions; their published speedups are not predictions for PR32. This bounded
literature check does not establish a comprehensive state-of-the-art or novelty
claim. A live arXiv API query for LLM/MPC work timed out and is not evidence that
no such work exists.

| Source | Relevant design lesson | What does not transfer automatically |
|---|---|---|
| Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*, OSDI 2024, https://www.usenix.org/conference/osdi24/presentation/agrawal | Prefill/decode have different batching behavior; chunking and schedule construction affect interference and latency | PR32 admission-budget control does not acquire Sarathi's scheduler actions or its reported hardware-dependent gains |
| Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, OSDI 2024, https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin | Goodput must account for phase-specific latency requirements; resource allocation changes the plant | Its multi-GPU disaggregation is not available on a single co-located PR32 GPU, and its joint TTFT/TPOT contract is not the existing TTFT-only primary contract |
| Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and Design*, second-edition official site, https://sites.engineering.ucsb.edu/~jbraw/mpc/ | Separate predictive optimization from the assumptions needed for feasibility/stability claims | A textbook terminal-set argument does not prove a partially observed serving plant stable under unknown EOS, unbounded arrivals or stalls |

The practical research contribution must therefore be falsifiable action
identification, coherent request accounting, measured observer/actuator timing,
and a fair closed-loop comparison. Merely labeling a learned predictor plus a
cap selector "MPC" is not a state-of-the-art result.

Two architecture hazards follow directly from the current source, independently
of a preferred MPC family:

1. `/metrics` supplies aggregates, not the complete marked request ledger.
   `Telemetry.active` presently retains receipt/admission/first-content state;
   emitted admission events have additional marks. A low-latency, coherent,
   versioned observer would be a new engineering deliverable, not an existing
   capability. Sampling and source availability must be measured.
2. The current `/admin/budget` has authentication and a versioned acknowledgement
   but no compare-and-set, deadline expiry or idempotent command contract.
   A client timeout cannot prove that a command did not or will not apply.
   Serializing client calls alone does not solve a delayed old command arriving
   after a new observation. Any live MPC adapter must qualify its server
   capabilities before sending actions and fail observably on ambiguity.

These are proposed controller-interface requirements, not additional reproduced
P2 regressions in the older state-envelope implementation.
