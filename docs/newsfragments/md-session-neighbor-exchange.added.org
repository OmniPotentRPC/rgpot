Added optional MD-session neighbour exchange slots to =rgpot_md_session_ops=
using the shared =rgpot_nlist_request= /=rgpot_nlist_soa= shape, with clear
unsupported status reporting.

The OpenMM session now serves neighbour queries from the current Context
geometry through the shared rgpot neighbour builder fallback and reports
external neighbour injection as unsupported.
