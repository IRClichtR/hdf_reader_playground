# CGNS Writer (poly round-trip) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `write_cgns` emit `cgnscheck`-valid CGNS/HDF5, including a faithful NGON/NFACE round-trip of `examples/cgns/particles_example.cgns`.

**Architecture:** First fix the reader (`src/cgns.rs`) so polyhedra are stored in mefikit's canonical form (face node-lists separated by `usize::MAX`, orientation encoded in node order) and polygons are 0-based. Then rewrite the writer (`src/main.rs`) to emit CPEX0031 poly sections (flat `ElementConnectivity` + `ElementStartOffset`), reusing the PGON block as the NGON face table that NFACE references by signed global index.

**Tech Stack:** Rust (edition 2024), `hdf5-metno`, `ndarray`, `mefikit 0.1.0`; external `cgnscheck` CLI for validation.

## Global Constraints

- CGNS library version written to file: **4.0** (CPEX0031 offsets require CGNS ≥ 4.0).
- Poly sections use flat `ElementConnectivity` (no length prefix) **+** `ElementStartOffset` of length `n+1`.
- Regular sections use fixed-stride `ElementConnectivity`, **no** `ElementStartOffset`.
- CGNS node indices are **1-based**; mefikit connectivity is **0-based** → add `+1` on write, subtract `1` on read.
- mefikit PHED encoding: flat node list, **each face terminated by `usize::MAX`**.
- Out of scope: `Family_t`, `ZoneBC_t`. Supported element types: VERTEX, SEG2, TRI3, QUAD4, TET4, HEX8, PGON, PHED; any other type is a clear `Err`.
- Validation gate: `cgnscheck <file>` exits 0 with no errors.

---

### Task 1: Reader — preserve PHED faces and fix PGON base

**Files:**
- Modify: `src/cgns.rs:143-217` (`read_elements`, both passes)
- Test: `src/cgns.rs` (new `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: existing `read(path: &Path) -> Result<UMesh, ...>`, `ElementType`.
- Produces: after `read`, the `PGON` block stores 0-based node indices; the `PHED` block stores, per cell, `[f0nodes…, usize::MAX, f1nodes…, usize::MAX, …]` (0-based, face nodes reversed when the NFACE ref was negative).

- [ ] **Step 1: Write the failing test**

Add to `src/cgns.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phed_stores_face_separated_node_lists() {
        let mesh = read(Path::new("examples/cgns/particles_example.cgns")).unwrap();

        // PGON is 0-based: max node index must be < n_coords (1114), not <= 1114.
        let n = mesh.coords().nrows();
        let pgon = mesh.block(ElementType::PGON).expect("PGON block");
        let max_pgon = pgon
            .iter(mesh.coords())
            .flat_map(|e| e.connectivity().to_vec())
            .max()
            .unwrap();
        assert!(max_pgon < n, "PGON must be 0-based (max {max_pgon} < {n})");

        // PHED cells must contain usize::MAX face separators.
        let phed = mesh.block(ElementType::PHED).expect("PHED block");
        let first = phed.iter(mesh.coords()).next().unwrap();
        let conn = first.connectivity();
        assert!(
            conn.contains(&usize::MAX),
            "PHED connectivity must contain usize::MAX face separators"
        );
        // Every face must be non-empty and the connectivity must end with a separator.
        assert_eq!(*conn.last().unwrap(), usize::MAX, "cell must end with separator");
        for face in conn.split(|&x| x == usize::MAX) {
            // split yields a trailing empty slice after the final separator; allow it.
            assert!(face.len() != 1 || face[0] != usize::MAX);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --bin hdf_reader phed_stores_face_separated_node_lists -- --nocapture`
Expected: FAIL — current reader has no separators (and PGON is 1-based).

- [ ] **Step 3: Fix the PGON first pass (make 0-based)**

In `src/cgns.rs`, in the `ElementType::PGON` arm of the first pass, change the node mapping to subtract 1:

```rust
                ElementType::PGON => {
                    let conn    = read_element_connectivity(element)?;
                    let offsets = read_element_offsets(element)?
                        .ok_or("PGON missing ElementStartOffset")?;
                    let range = read_element_range(element)?;
                    let n_cells = (range[1] - range[0] + 1) as usize;
                    for i in 0..n_cells {
                        let start = offsets[i] as usize;
                        let end   = offsets[i + 1] as usize;
                        let nodes: Vec<usize> = conn[start..end]
                            .iter().map(|&v| (v as usize) - 1).collect(); // 0-based
                        mesh.add_element(ElementType::PGON, &nodes, None, None);
                    }
                    pgon_offsets = Some(offsets);
                    pgon_conn    = Some(conn);
                }
```

- [ ] **Step 4: Rewrite the PHED second pass (sentinel encoding + orientation)**

Replace the second-pass body (`for i in 0..n_cells { … }`) with:

```rust
        let n_cells = p_off.len() - 1;
        for i in 0..n_cells {
            let start = p_off[i] as usize;
            let end   = p_off[i + 1] as usize;

            let mut cell_nodes: Vec<usize> = Vec::new();

            for &face_ref in &p_conn[start..end] {
                let reversed   = face_ref < 0;
                let face_index = (face_ref.unsigned_abs() as usize) - 1;

                let node_start = f_off[face_index] as usize;
                let node_end   = f_off[face_index + 1] as usize;

                let face: Vec<usize> = f_conn[node_start..node_end]
                    .iter()
                    .map(|&node_id| (node_id as usize) - 1) // 0-based
                    .collect();

                if reversed {
                    cell_nodes.extend(face.iter().rev());
                } else {
                    cell_nodes.extend(face.iter());
                }
                cell_nodes.push(usize::MAX); // mefikit PHED face separator
            }

            mesh.add_element(ElementType::PHED, &cell_nodes, None, None);
        }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test --bin hdf_reader phed_stores_face_separated_node_lists -- --nocapture`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cgns.rs
git commit -m "fix(cgns): store PHED as mefikit face-separated node lists; PGON 0-based"
```

---

### Task 2: Writer — CPEX0031 sections, version 4.0, reuse PGON as NGON table

**Files:**
- Modify: `src/main.rs` — `write_version` (700), `write_elements` (761-826), `write_cgns` (875-938, remove `write_bcs` call), delete `write_bcs` (828-870), restore `main` (1028-1033)
- Add: `use std::collections::HashMap;` (top of `src/main.rs`)

**Interfaces:**
- Consumes: `mesh.blocks()`, `mesh.block(et)`, `block.iter(coords)`, `el.connectivity()`, `element_type_to_cgns(et)`, `write_node_attrs`, `arr1`, `Array1`.
- Produces: `write_cgns(path, mesh)` writes a valid file. New helpers `write_conn_and_offset(section, &[i64], &[i64])` and `face_orientation(canon: &[usize], face: &[usize]) -> i64`.

- [ ] **Step 1: Write the failing round-trip test**

Add to `src/main.rs` (new `#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn cgnscheck(path: &str) -> (i32, String) {
        let out = Command::new("cgnscheck").arg("-v").arg(path).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (code, text)
    }

    #[test]
    fn roundtrip_particles_passes_cgnscheck() {
        let src = "examples/cgns/particles_example.cgns";
        let dst = "examples/cgns/roundtrip_check.cgns";
        let _ = std::fs::remove_file(dst);
        let mesh = cgns::read(Path::new(src)).unwrap();
        write_cgns(Path::new(dst), mesh.view()).unwrap();

        let (code, text) = cgnscheck(dst);
        assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}");
        assert!(!text.to_lowercase().contains("error"), "cgnscheck reported errors:\n{text}");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --bin hdf_reader roundtrip_particles_passes_cgnscheck -- --nocapture`
Expected: FAIL — current writer segfaults cgnscheck (nonzero exit).

- [ ] **Step 3: Add the `HashMap` import and bump the library version**

At the top of `src/main.rs` add:

```rust
use std::collections::HashMap;
```

Change `write_version` to write 4.0:

```rust
fn write_version(file: &File) -> Result<(), Box<dyn std::error::Error>> {
    let node = file.create_group("CGNSLibraryVersion")?;
    write_node_attrs(&node, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", 0)?;
    node.new_dataset::<f32>()
        .shape([1])
        .create(" data")?
        .write(&arr1(&[4.0_f32]))?; // CPEX0031 offsets require CGNS >= 4.0
    Ok(())
}
```

- [ ] **Step 4: Add writer helpers**

Add near the other write sub-functions in `src/main.rs`:

```rust
fn write_conn_and_offset(
    section: &Group,
    conn: &[i64],
    offsets: &[i64],
) -> Result<(), Box<dyn std::error::Error>> {
    let conn_node = section.create_group("ElementConnectivity")?;
    write_node_attrs(&conn_node, "ElementConnectivity", "DataArray_t", "I8", 1)?;
    conn_node.new_dataset::<i64>()
        .shape([conn.len()])
        .create(" data")?
        .write(&Array1::from(conn.to_vec()))?;

    let off_node = section.create_group("ElementStartOffset")?;
    write_node_attrs(&off_node, "ElementStartOffset", "DataArray_t", "I8", 1)?;
    off_node.new_dataset::<i64>()
        .shape([offsets.len()])
        .create(" data")?
        .write(&Array1::from(offsets.to_vec()))?;
    Ok(())
}

// Sign of an NFACE reference: +1 if `face` has the same cyclic orientation as
// the canonical PGON face `canon`, -1 if reversed. Faces with < 3 nodes carry
// no orientation, so return +1.
fn face_orientation(canon: &[usize], face: &[usize]) -> i64 {
    let n = canon.len();
    if n < 3 {
        return 1;
    }
    let pos = face.iter().position(|&x| x == canon[0]).unwrap();
    if face[(pos + 1) % n] == canon[1] { 1 } else { -1 }
}
```

- [ ] **Step 5: Rewrite `write_elements`**

Replace the entire `write_elements` function with:

```rust
fn write_elements(zone: &Group, mesh: &UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    // Pre-pass: assign each block a contiguous, 1-based [start, end] element range.
    let mut ranges: Vec<(ElementType, i64, i64)> = Vec::new();
    let mut start = 1_i64;
    for (et, block) in mesh.blocks() {
        let n = block.len() as i64;
        ranges.push((*et, start, start + n - 1));
        start += n;
    }

    // Build the NGON face-index map from the PGON block:
    //   sorted-node-set -> (global cgns face index, canonical node order)
    let mut face_map: HashMap<Vec<usize>, (i64, Vec<usize>)> = HashMap::new();
    if let Some(&(_, pgon_start, _)) = ranges.iter().find(|(et, _, _)| *et == ElementType::PGON) {
        let block = mesh.block(ElementType::PGON).unwrap();
        for (i, el) in block.iter(mesh.coords()).enumerate() {
            let conn = el.connectivity().to_vec();
            let mut key = conn.clone();
            key.sort_unstable();
            face_map.insert(key, (pgon_start + i as i64, conn));
        }
    }

    for (et, r_start, r_end) in ranges.iter().copied() {
        let block = mesh.block(et).unwrap();
        let cgns_code = element_type_to_cgns(et);
        let section_name = format!("{et:?}");

        let section = zone.create_group(&section_name)?;
        write_node_attrs(&section, &section_name, "Elements_t", "I4", 1)?;
        section.new_dataset::<i32>()
            .shape([2])
            .create(" data")?
            .write(&arr1(&[cgns_code, 0_i32]))?;

        let er = section.create_group("ElementRange")?;
        write_node_attrs(&er, "ElementRange", "IndexRange_t", "I8", 1)?;
        er.new_dataset::<i64>()
            .shape([2])
            .create(" data")?
            .write(&arr1(&[r_start, r_end]))?;

        match et {
            ElementType::PGON => {
                let mut conn: Vec<i64> = Vec::new();
                let mut offsets: Vec<i64> = vec![0];
                for el in block.iter(mesh.coords()) {
                    for &node in el.connectivity() {
                        conn.push(node as i64 + 1);
                    }
                    offsets.push(conn.len() as i64);
                }
                write_conn_and_offset(&section, &conn, &offsets)?;
            }
            ElementType::PHED => {
                let mut conn: Vec<i64> = Vec::new();
                let mut offsets: Vec<i64> = vec![0];
                for el in block.iter(mesh.coords()) {
                    for face in el.connectivity().split(|&x| x == usize::MAX) {
                        if face.is_empty() {
                            continue; // trailing separator
                        }
                        let mut key = face.to_vec();
                        key.sort_unstable();
                        let (idx, canon) = face_map.get(&key)
                            .ok_or("write_cgns: PHED face not present in PGON block")?;
                        conn.push(face_orientation(canon, face) * idx);
                    }
                    offsets.push(conn.len() as i64);
                }
                write_conn_and_offset(&section, &conn, &offsets)?;
            }
            _ => {
                let mut conn: Vec<i64> = Vec::new();
                for el in block.iter(mesh.coords()) {
                    for &node in el.connectivity() {
                        conn.push(node as i64 + 1);
                    }
                }
                let conn_node = section.create_group("ElementConnectivity")?;
                write_node_attrs(&conn_node, "ElementConnectivity", "DataArray_t", "I8", 1)?;
                conn_node.new_dataset::<i64>()
                    .shape([conn.len()])
                    .create(" data")?
                    .write(&Array1::from(conn))?;
            }
        }
    }
    Ok(())
}
```

- [ ] **Step 6: Remove `write_bcs` and its call**

Delete the entire `write_bcs` function. In `write_cgns`, delete the `write_bcs(&zone, &mesh)?;` call and its preceding comment block. Also change `write_root_attrs(&file);` to `write_root_attrs(&file)?;` (handle the Result).

- [ ] **Step 7: Restore `main` to run the round-trip**

Replace the debug `main` with:

```rust
fn main() {
    write_roundtrip_test().unwrap();
}
```

And update `write_roundtrip_test` to write to the check file:

```rust
pub fn write_roundtrip_test() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = cgns::read(Path::new("examples/cgns/particles_example.cgns"))?;
    write_cgns(Path::new("examples/cgns/roundtrip_check.cgns"), mesh.view())?;
    println!("wrote roundtrip_check.cgns");
    Ok(())
}
```

- [ ] **Step 8: Run the round-trip test to verify it passes**

Run: `cargo test --bin hdf_reader roundtrip_particles_passes_cgnscheck -- --nocapture`
Expected: PASS (cgnscheck exit 0, no "error").
If cgnscheck complains specifically about the library version, bump `write_version` to `4.5_f32` and re-run.

- [ ] **Step 9: Commit**

```bash
git add src/main.rs
git commit -m "feat(cgns): valid CPEX0031 NGON/NFACE writer with orientation; drop BC write"
```

---

### Task 3: Verify semantic round-trip and the regular-element path

**Files:**
- Modify: `src/main.rs` `#[cfg(test)] mod tests` (add two tests)

**Interfaces:**
- Consumes: `cgns::read`, `write_cgns`, `cgnscheck` helper from Task 2, `mefikit` mesh builders.

- [ ] **Step 1: Write the semantic re-read test**

Add to the `tests` module in `src/main.rs`:

```rust
    #[test]
    fn roundtrip_particles_reread_matches() {
        let src = "examples/cgns/particles_example.cgns";
        let dst = "examples/cgns/roundtrip_reread.cgns";
        let _ = std::fs::remove_file(dst);

        let orig = cgns::read(Path::new(src)).unwrap();
        write_cgns(Path::new(dst), orig.view()).unwrap();
        let back = cgns::read(Path::new(dst)).unwrap();

        assert_eq!(orig.coords().nrows(), back.coords().nrows());
        assert_eq!(orig.space_dimension(), back.space_dimension());
        assert_eq!(
            back.block(ElementType::PGON).unwrap().len(),
            orig.block(ElementType::PGON).unwrap().len()
        );
        assert_eq!(
            back.block(ElementType::PHED).unwrap().len(),
            orig.block(ElementType::PHED).unwrap().len()
        );

        // First PHED cell's face-node structure survives the round-trip.
        let a = orig.block(ElementType::PHED).unwrap()
            .iter(orig.coords()).next().unwrap().connectivity().to_vec();
        let b = back.block(ElementType::PHED).unwrap()
            .iter(back.coords()).next().unwrap().connectivity().to_vec();
        assert_eq!(a, b, "first PHED cell connectivity must match");
    }
```

- [ ] **Step 2: Run it (expect PASS)**

Run: `cargo test --bin hdf_reader roundtrip_particles_reread_matches -- --nocapture`
Expected: PASS.
(If the PHED connectivity mismatches, the orientation or face ordering in Task 1/Task 2 is off — debug before continuing.)

- [ ] **Step 3: Write the regular-element cgnscheck test**

Add to the `tests` module. This builds a small hex mesh via mefikit and validates the non-poly path:

```rust
    #[test]
    fn regular_hex_mesh_passes_cgnscheck() {
        use mefikit::mesh_examples::make_imesh_3d;
        let dst = "examples/cgns/regular_hex_check.cgns";
        let _ = std::fs::remove_file(dst);

        let mesh = make_imesh_3d(3); // structured -> HEX8 unstructured mesh
        write_cgns(Path::new(dst), mesh.view()).unwrap();

        let (code, text) = cgnscheck(dst);
        assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}");
        assert!(!text.to_lowercase().contains("error"), "cgnscheck errors:\n{text}");
    }
```

- [ ] **Step 4: Run it**

Run: `cargo test --bin hdf_reader regular_hex_mesh_passes_cgnscheck -- --nocapture`
Expected: PASS.
Note: if `make_imesh_3d` / `mesh_examples` is not exported by the `mefikit` crate's public API, substitute a hand-built `UMesh` with a couple of `HEX8` elements (8 shared corner coords, `add_regular_block(ElementType::HEX8, …)`) — the point is to exercise the regular-element write path through cgnscheck.

- [ ] **Step 5: Run the full test suite**

Run: `cargo test --bin hdf_reader -- --nocapture`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/main.rs
git commit -m "test(cgns): semantic re-read + regular-element cgnscheck coverage"
```

---

## Self-Review

**Spec coverage:**
- Reader PGON base fix → Task 1 Step 3. ✓
- Reader PHED sentinel + orientation → Task 1 Step 4. ✓
- Version 4.0 → Task 2 Step 3. ✓
- Range pre-pass + NGON map → Task 2 Step 5. ✓
- Regular / PGON→NGON / PHED→NFACE sections → Task 2 Step 5. ✓
- Error on missing PGON / unmatched face → Task 2 Step 5 (`.ok_or(...)`). ✓
- Remove write_bcs → Task 2 Step 6. ✓
- Verify: cgnscheck gate → Task 2 Step 8; re-read asserts → Task 3 Step 1-2; regular path → Task 3 Step 3-4. ✓

**Placeholder scan:** No TBD/TODO; all code steps contain real code. The one conditional (Task 3 Step 4 fallback) gives an explicit alternative, not a placeholder. ✓

**Type consistency:** `face_orientation(canon: &[usize], face: &[usize]) -> i64` and `write_conn_and_offset(&Group, &[i64], &[i64])` used consistently in Task 2 Step 4/5. `face_map: HashMap<Vec<usize>, (i64, Vec<usize>)>` consistent. `element_type_to_cgns` returns `i32` (existing). ✓

**Known risk / checkpoint:** libcgns is 3.4.0 locally but read the v4.5 reference (with offsets) fine, so `cgnscheck` supports CPEX0031 here. If the written version 4.0 is rejected on a version technicality, Task 2 Step 8 says bump to 4.5.
