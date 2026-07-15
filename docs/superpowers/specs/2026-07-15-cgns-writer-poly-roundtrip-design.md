# CGNS writer with poly (NGON/NFACE) round-trip — design

Date: 2026-07-15
Status: approved (design), pending spec review

## Problem

`write_cgns` in `src/main.rs` produces CGNS files that `cgnscheck` rejects — it
**segfaults** (exit 139) while reading the poly element sections of a generated
file. The goal is a correct writer that turns a mefikit `UMeshView` into a
`cgnscheck`-valid CGNS/HDF5 file, including a faithful round-trip of the
polygon/polyhedron (NGON/NFACE) mesh in `examples/cgns/particles_example.cgns`.

Out of scope (explicit): `Family_t` and `ZoneBC_t` are not written.

## Root cause (investigation summary)

- The valid CGNS encoding for `NGON_n` / `NFACE_n` (confirmed from the real
  `particles_example.cgns`) is **CPEX0031**: `ElementConnectivity` is a *flat*
  array with **no length prefix**, paired with an `ElementStartOffset` array of
  length `n+1`. The old writer emitted the *legacy* prefixed format
  (`[k, v0..vk, …]`) and **no `ElementStartOffset`**, and declared library
  version 3.4 (which predates CPEX0031). That mismatch is what crashes readers.
- Deeper blocker: the reader in `src/cgns.rs` resolved each `NFACE` polyhedron
  into a **flat, concatenated list of node indices with face boundaries and
  orientation discarded**. A valid `NFACE_n` section references *faces* (signed,
  1-based face-element indices), which cannot be reconstructed from that.
- mefikit has a **canonical polyhedron encoding** (`element_traits/element_topo.rs:155`,
  the `PHED` arm of `to_subentities`): a flat node list where **each face is
  terminated by a `usize::MAX` sentinel**
  (`co.split_inclusive(|&e| e == usize::MAX)`). The current reader does *not*
  emit these sentinels, so it is wrong by mefikit's own model.
- Secondary bug: the reader stores `PGON` face nodes **1-based**
  (`v as usize`, no `-1`), inconsistent with mefikit's 0-based node convention.

## In-memory facts (measured)

Reading `particles_example.cgns` with the current reader yields:

- `space_dim = 3`, `top_dim = D3`, `n_coords = 1114`.
- `PGON` block: 2583 elements, each a node-index list (currently 1-based).
- `PHED` block: 310 elements, each a flat concatenated node list (0-based, no
  face separators, orientation lost), e.g. lengths 30/36/36.

Reference file section layout (ground truth):

- `CELL_FACES` = `NGON_n`, ` data = [22, 0]` (I4), `ElementRange = [1, 2583]`,
  `ElementConnectivity` (I8, flat, 8351), `ElementStartOffset` (I8, 2584).
- `CELLS` = `NFACE_n`, ` data = [23, 0]` (I4), `ElementRange = [2584, 2893]`,
  `ElementConnectivity` (I8, flat, 4292, signed face refs incl. negatives),
  `ElementStartOffset` (I8, 311).

## Design

### 1. Reader change — `src/cgns.rs`, `read_elements`

**PGON base fix.** In the first pass, store faces 0-based: map `(v - 1)` instead
of `v` before `mesh.add_element(ElementType::PGON, …)`. (The second pass already
reads the raw, still-1-based `pgon_conn` buffer and subtracts 1 there; that path
is unaffected because it uses the raw read, not the mesh copy.)

**PHED preserve-faces.** Replace the second-pass body that builds a flat
`cell_nodes` of resolved nodes with mefikit's canonical encoding. For each cell,
for each signed `face_ref` in `p_conn[start..end]`:

- `reversed = face_ref < 0`
- `face_index = face_ref.unsigned_abs() as usize - 1`
- `node_start = f_off[face_index]`, `node_end = f_off[face_index + 1]`
- collect the face's nodes `f_conn[node_start..node_end]` as 0-based
  (`node_id - 1`)
- if `reversed`, push them **reversed**; else in order
- push `usize::MAX` as the face separator

The resulting cell connectivity is
`[f0n0,…, MAX, f1n0,…, MAX, …, fKn0,…, MAX]`. This preserves face structure and
encodes orientation in node order, and is consumable by mefikit's PHED topology
code. Add it with `mesh.add_element(ElementType::PHED, &cell_nodes, None, None)`.

### 2. Writer — `src/main.rs`

Replace the current write path (`write_elements`, `write_bcs`, and the poly
encoding). Keep the existing, working node primitives: `write_nullterm_str_attr`,
`write_node_attrs`, `write_root_attrs`, `write_c1_data`, `write_version`,
`write_base`, `write_zone`, `write_coords`, plus the ` format` / ` hdf5version`
datasets and `write_cgns`'s scaffolding.

Changes:

- **Library version:** write `CGNSLibraryVersion` = **4.0** (CPEX0031 requires
  CGNS ≥ 4.0). Update `write_version` to `4.0_f32`.
- **Range pre-pass:** before writing sections, iterate `mesh.blocks()` once to
  assign each block a contiguous 1-based `[range_start, range_end]`. This makes
  the NGON face-index map correct independent of iteration order.
- **PGON face-index map:** if a `PGON` block exists, build
  `map: HashMap<Vec<usize> /*sorted node set*/, i64 /*global cgns index*/>` while
  (or before) writing its `NGON_n` section, using the block's assigned
  `range_start`. Key is the face's node indices **sorted ascending**; value is
  the face's global CGNS element index.
- **Regular blocks** (VERTEX, SEG2, TRI3, QUAD4, TET4, HEX8): standard
  `Elements_t` section:
  - ` data = [cgns_code, 0]` (I4),
  - `ElementRange` (I8, `[range_start, range_end]`),
  - `ElementConnectivity` (I8, fixed-stride, each node `+1`). No
    `ElementStartOffset`.
- **PGON → `NGON_n` (code 22):**
  - ` data = [22, 0]`,
  - `ElementRange`,
  - `ElementConnectivity` (I8): flat node indices `+1`, no length prefix,
  - `ElementStartOffset` (I8): length `n+1`, prefix sums of per-face node counts.
- **PHED → `NFACE_n` (code 23):**
  - ` data = [23, 0]`,
  - `ElementRange`,
  - for each cell, split its connectivity on `usize::MAX` to recover faces; for
    each face compute its sorted node-set, look it up in the PGON map to get the
    global face index `idx`; determine the sign by comparing the cell face's
    node order to the PGON face's node order (matching cyclic order → `+idx`,
    reversed → `-idx`); emit signed refs,
  - `ElementConnectivity` (I8): flat signed face refs,
  - `ElementStartOffset` (I8): length `n+1`, prefix sums of per-cell face counts.
- **Errors:** if a `PHED` block exists with no `PGON` block, or a PHED face's
  node-set is not found in the map, return a descriptive `Err` (documented
  limitation of the "reuse PGON block" strategy).
- **Remove `write_bcs`** and its call — `Family_t`/`ZoneBC_t` are out of scope.

Sign determination detail: two faces with the same sorted node-set match; the
PGON face has a canonical node order `p`. The cell face order `c` is either a
rotation of `p` (same orientation → `+`) or a rotation of `reverse(p)` (opposite
→ `-`). Compare by finding `p[0]` in `c` and checking the direction of the next
node.

### 3. Verification

1. `cargo run`: `particles_example.cgns` → `cgns::read` → `write_cgns` to a new
   file (e.g. `examples/cgns/roundtrip_check.cgns`).
2. **Gate:** `cgnscheck -v <newfile>` completes with **no errors** (exit 0).
3. Re-read the written file with `cgns::read`; assert:
   - block set = {PGON, PHED}, counts 2583 / 310,
   - `n_coords = 1114`, `space_dim = 3`, `top_dim = D3`,
   - spot-check that a sample of PGON and PHED connectivities equals the
     original in-memory mesh.
4. Regular-element path: build a mefikit `mesh_examples` mesh (e.g. QUAD4 or a
   HEX imesh) and run it through `write_cgns` → `cgnscheck` (exit 0).

## Non-goals

- `Family_t`, `ZoneBC_t`, flow solutions, iterative/base data, dimensional
  units.
- Element types outside {VERTEX, SEG2, TRI3, QUAD4, TET4, HEX8, PGON, PHED};
  others produce a clear error.
- Byte-for-byte reproduction of the source file (structure and semantics match;
  auxiliary nodes are not reproduced).
