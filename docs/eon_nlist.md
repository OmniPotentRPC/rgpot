# Calling the neighbour list from eOn

eOn potentials that today call `vesin_neighbors` can use the same request
through the C ABI or through Cap'n Proto. Both speak the columns
`first`, `second`, `shift`, `vector`, `distance`. Indices are
zero-based. The vector is `r_j - r_i + S H`, which is the layout a
Fortran `(3, npairs)` array already has.

## C ABI

A fresh list is two calls: one to learn the length, one to fill caller
buffers. A skin reuses an `rgpot_nlist_builder`. The cut has to be loose
when the skin is positive, because a strict cut stores no buffer.

```c
rgpot_nlist_request request;
rgpot_nlist_request_init(&request);
request.cutoff = 5.0;
request.skin = 0.5;
request.cut = 1;          /* loose: candidates out to cutoff + skin */
request.shell = 1;        /* full list, both directions */
request.want_shifts = 1;
request.want_vectors = 1;
request.want_distances = 1;

rgpot_nlist_builder *builder = NULL;
rgpot_nlist_builder_open(&builder);

size_t needed = 0;
rgpot_nlist_builder_update(builder, positions, natoms, box9, &request,
                           NULL, NULL, &needed, NULL);

/* allocate first[needed], second[needed], shift[3*needed],
   vector[3*needed], distance[needed] */

rgpot_nlist_soa soa = {0};
soa.bytes = (uint32_t)sizeof(soa);
soa.first = first;
soa.second = second;
soa.shift = shift;
soa.vector = vector;
soa.distance = distance;
soa.cap = needed;

int32_t reused = 0;
rgpot_nlist_builder_update(builder, positions, natoms, box9, &request,
                           &soa, &needed, &needed, &reused);
rgpot_nlist_builder_close(builder);
```

`reused == 1` means every atom was still within half the skin, so the
pair keys were kept. The Fortran gather then rebuilds `r_j - r_i + S H`
from those keys and the positions of this call, and drops pairs that
are no longer inside the cutoff. That is `rgpot_nlist_fortran_build`.

## Cap'n Proto

The same request is one `NeighborRequest`. `format = 1` is a full list,
`cut = 1` is loose, and `skin` is the Verlet buffer. The reply is a
`NeighborList` with those columns. A client that already streams chunks
sends `cursor` and reads `next` until `more` is false.

```text
(cutoff = 5, skin = 0.5, format = 1, cut = 1,
 shifts = true, vectors = true, distances = true, mode = 1)
```
