use crate::read_type_attr2;
use hdf5_metno::types::FixedAscii;
use hdf5_metno::{File, Group};
use mefikit::mesh::{ElementType, UMesh};
use std::path::Path;

// The cgns module reads CGNS files. It is strictly limited to the CGNS/HDF5
// format since it uses hdf5-metno as the interface to the HDF5 library.
//
// Ported from mefikit's `io/cgns_io.rs` structure, but with the correct
// polyhedron handling:
//   * PGON node indices are stored 0-based (mefikit convention).
//   * PHED cells are stored as mefikit-native face node-lists separated by
//     `usize::MAX`, with face orientation encoded in node order (reversed when
//     the NFACE reference is negative). This preserves face structure so
//     mefikit's own polyhedron topology/geometry code works on the mesh.
//
// Family_t and ZoneBC_t are intentionally not read (future work).

// ── element type mapping ──────────────────────────────────────────────────────

// Minimal local stand-in for mefikit's `ElementsMapping` (which is not part of
// mefikit 0.1.0). Maps CGNS element type codes to `ElementType`.
struct ElementsMapping {
    _name: &'static str,
    entries: &'static [(u32, ElementType)],
}

impl ElementsMapping {
    const fn new(name: &'static str, entries: &'static [(u32, ElementType)]) -> Self {
        Self { _name: name, entries }
    }

    fn to_element(&self, code: u32) -> Result<ElementType, Box<dyn std::error::Error>> {
        for &(c, et) in self.entries {
            if c == code {
                return Ok(et);
            }
        }
        Err(format!("unsupported CGNS element type code {code}").into())
    }
}

// CGNS element type codes (see the CGNS SIDS ElementType_t enumeration). Poly
// types (NGON_n/NFACE_n) have variable stride and are handled separately; their
// node count comes from `ElementType::num_nodes()` returning `None`.
const CGNS_MAPPING: ElementsMapping = ElementsMapping::new(
    "CGNS",
    &[
        (2, ElementType::VERTEX),
        (3, ElementType::SEG2),
        (5, ElementType::TRI3),
        (7, ElementType::QUAD4),
        (10, ElementType::TET4),
        (17, ElementType::HEX8),
        (22, ElementType::PGON),
        (23, ElementType::PHED),
    ],
);

// ── base dimensions ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CgnsBaseDim {
    pub cell_dim: usize,
    pub phys_dim: usize,
}

impl TryFrom<&Group> for CgnsBaseDim {
    type Error = Box<dyn std::error::Error>;

    fn try_from(base: &Group) -> Result<Self, Self::Error> {
        let data: Vec<i32> = base
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i32>()?
            .into_raw_vec_and_offset()
            .0;

        if data.len() < 2 {
            return Err("CGNSBase_t data must have at least 2 elements".into());
        }

        let base_data = Self {
            cell_dim: data[0] as usize,
            phys_dim: data[1] as usize,
        };

        base_data.validate()?;
        Ok(base_data)
    }
}

impl CgnsBaseDim {
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        match (self.cell_dim, self.phys_dim) {
            (3, 3) | (2, 3) | (2, 2) | (1, 1) => Ok(()),
            other => Err(format!("Unsupported dimension combo {other:?}").into()),
        }
    }
}

// ── node/label helpers ────────────────────────────────────────────────────────

pub fn cgns_label(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr(" label").or_else(|_| group.attr("label"))?;
    let label: String = attr
        .as_reader()
        .read_scalar::<FixedAscii<64>>()?
        .to_string();
    Ok(label.trim().trim_matches('\0').to_string())
}

pub fn find_first_child_with_label(
    group: &Group,
    label: &str,
) -> Result<Group, Box<dyn std::error::Error>> {
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        if lbl == label {
            return Ok(child);
        }
    }
    Err(format!("no child with label '{label}' in '{}'", group.name()).into())
}

pub fn children_with_label(
    group: &Group,
    label: &str,
) -> Result<Vec<Group>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        if lbl == label {
            out.push(child);
        }
    }
    Ok(out)
}

pub fn read_string_data(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let s: String = group
        .dataset(" data")?
        .as_reader()
        .read_1d::<i8>()?
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8 as char)
        .collect();
    Ok(s.trim().to_string())
}

// ── coordinates ───────────────────────────────────────────────────────────────

pub fn read_coordinates(
    zone: &Group,
    phys_dim: usize,
) -> Result<ndarray::ArcArray2<f64>, Box<dyn std::error::Error>> {
    let gc = find_first_child_with_label(zone, "GridCoordinates_t")?;
    let names = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    let columns: Vec<Vec<f64>> = (0..phys_dim)
        .map(|i| {
            // CoordinateX/Y/Z are groups; data lives in their " data" dataset.
            let coord_group = gc.group(names[i])?;
            let ds = coord_group.dataset(" data")?;

            // handle both R4 and R8 — check the "type" attribute
            let type_attr = coord_group
                .attr("type")
                .and_then(|a| a.as_reader().read_scalar::<FixedAscii<8>>())
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "R8".to_string());

            let values: Vec<f64> = if type_attr.trim_matches('\0').starts_with("R4") {
                ds.as_reader().read_1d::<f32>()?.iter().map(|&v| v as f64).collect()
            } else {
                ds.as_reader().read_1d::<f64>()?.to_vec()
            };

            Ok(values)
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    let n = columns[0].len();
    let mut coords = ndarray::Array2::<f64>::zeros((n, phys_dim));
    for (col_idx, col_data) in columns.iter().enumerate() {
        coords
            .column_mut(col_idx)
            .iter_mut()
            .zip(col_data)
            .for_each(|(dst, &src)| *dst = src);
    }

    Ok(coords.into_shared())
}

// ── element sub-readers ───────────────────────────────────────────────────────

fn read_index_array(group: &Group) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let ds = group.dataset(" data")?;
    let type_str = read_type_attr2(group, "type")?;

    let values: Vec<i64> = match type_str.as_str() {
        "I4" => ds.as_reader().read_dyn::<i32>()?.iter().map(|&v| v as i64).collect(),
        "I8" => ds.as_reader().read_dyn::<i64>()?.into_raw_vec_and_offset().0,
        other => return Err(format!("Unexpected index type: {other}").into()),
    };
    Ok(values)
}

// Elements_t stores [ElementType, ElementSizeBoundary] in its " data" dataset;
// the first value is the CGNS element type code.
fn read_element_type(element: &Group) -> Result<i32, Box<dyn std::error::Error>> {
    let data: Vec<i32> = element
        .dataset(" data")?
        .as_reader()
        .read_dyn::<i32>()?
        .into_raw_vec_and_offset()
        .0;
    data.first()
        .copied()
        .ok_or_else(|| format!("Elements_t '{}' has empty data", element.name()).into())
}

fn read_element_range(element: &Group) -> Result<[i64; 2], Box<dyn std::error::Error>> {
    let range_group = element.group("ElementRange")?;
    let values = read_index_array(&range_group)?;
    Ok([values[0], values[1]])
}

fn read_element_connectivity(element: &Group) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let conn_group = element.group("ElementConnectivity")?;
    read_index_array(&conn_group)
}

fn read_phed_connectivity(element: &Group) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let conn_group = element.group("ElementConnectivity")?;
    // PHED can contain negative values, so subtracting here is irrelevant.
    read_index_array(&conn_group)
}

fn read_element_offsets(element: &Group) -> Result<Option<Vec<i64>>, Box<dyn std::error::Error>> {
    let Ok(offset_group) = element.group("ElementStartOffset") else {
        return Ok(None);
    };
    let values = read_index_array(&offset_group)?;
    Ok(Some(values))
}

// CGNS defines two "polyhedral" element types (CPEX0031): NGON_n (code 22) holds
// polygon faces and NFACE_n (code 23) holds polyhedral cells that reference those
// faces by signed 1-based index. Both use a flat ElementConnectivity plus an
// ElementStartOffset of length n+1.
//
// ElementStartOffset = [0, 4, 9, 13, ...]
//                       ↑  ↑  ↑   ↑
//                       |  |  |   cell 3 starts at index 13
//                       |  |  cell 2 starts at index 9
//                       |  cell 1 starts at index 4
//                       cell 0 starts at index 0
fn read_elements(mesh: &mut UMesh, zone: &Group) -> Result<(), Box<dyn std::error::Error>> {
    let el_group = children_with_label(zone, "Elements_t")?;

    // --- first pass: add regular elements and PGON faces; buffer PHED raw data ---
    let mut pgon_offsets: Option<Vec<i64>> = None;
    let mut pgon_conn: Option<Vec<i64>> = None;
    let mut phed_offsets: Option<Vec<i64>> = None;
    let mut phed_conn: Option<Vec<i64>> = None;

    for element in &el_group {
        let code = read_element_type(element)?;
        // Unsupported sections are skipped with a warning rather than aborting.
        let element_type = match CGNS_MAPPING.to_element(code as u32) {
            Ok(et) => et,
            Err(_) => {
                eprintln!("warning: unsupported CGNS element type {code}, section skipped");
                continue;
            }
        };

        match element_type {
            ElementType::PGON => {
                let conn = read_element_connectivity(element)?;
                let offsets = read_element_offsets(element)?
                    .ok_or("PGON missing ElementStartOffset")?;
                let range = read_element_range(element)?;
                let n_cells = (range[1] - range[0] + 1) as usize;
                for i in 0..n_cells {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    let nodes: Vec<usize> = conn[start..end]
                        .iter()
                        .map(|&v| (v as usize) - 1) // 0-based
                        .collect();
                    mesh.add_element(ElementType::PGON, &nodes, None, None);
                }
                pgon_offsets = Some(offsets);
                pgon_conn = Some(conn);
            }
            ElementType::PHED => {
                phed_conn = Some(read_phed_connectivity(element)?);
                phed_offsets = Some(
                    read_element_offsets(element)?
                        .ok_or("PHED missing ElementStartOffset")?,
                );
            }
            other => {
                let range = read_element_range(element)?;
                let conn = read_element_connectivity(element)?;
                let n_cells = (range[1] - range[0] + 1) as usize;
                let nodes_per_cell = other.num_nodes().ok_or_else(|| {
                    format!("CGNS element type {other:?} has no fixed node count")
                })?;
                for i in 0..n_cells {
                    let start = i * nodes_per_cell;
                    let end = start + nodes_per_cell;
                    let cell: Vec<usize> = conn[start..end]
                        .iter()
                        .map(|&v| (v as usize) - 1) // 0-based
                        .collect();
                    mesh.add_element(other, &cell, None, None);
                }
            }
        }
    }

    // --- second pass: resolve PHED using PGON, preserving faces + orientation ---
    if let (Some(p_off), Some(p_conn), Some(f_off), Some(f_conn)) =
        (phed_offsets, phed_conn, pgon_offsets, pgon_conn)
    {
        let n_cells = p_off.len() - 1;
        for i in 0..n_cells {
            let start = p_off[i] as usize;
            let end = p_off[i + 1] as usize;

            let mut cell_nodes: Vec<usize> = Vec::new();

            for &face_ref in &p_conn[start..end] {
                let reversed = face_ref < 0;
                let face_index = (face_ref.unsigned_abs() as usize) - 1;

                let node_start = f_off[face_index] as usize;
                let node_end = f_off[face_index + 1] as usize;

                let face: Vec<usize> = f_conn[node_start..node_end]
                    .iter()
                    .map(|&node_id| (node_id as usize) - 1) // 0-based
                    .collect();

                // Orientation is encoded in node order: reverse the face's nodes
                // when the NFACE reference is negative (inward normal).
                if reversed {
                    cell_nodes.extend(face.iter().rev());
                } else {
                    cell_nodes.extend(face.iter());
                }
                cell_nodes.push(usize::MAX); // mefikit PHED face separator
            }

            mesh.add_element(ElementType::PHED, &cell_nodes, None, None);
        }
    }

    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

// DISCLAIMER: Family_t and ZoneBC_t are not read in this version. The code is
// structured to allow future implementation. The current implementation reads
// the mesh geometry and element connectivity, specifically unstructured meshes
// with regular, PGON and PHED elements.
pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    let base = find_first_child_with_label(&f.as_group()?, "CGNSBase_t")?;
    let cgns_dim = CgnsBaseDim::try_from(&base)?;

    let zone = find_first_child_with_label(&base, "Zone_t")?;

    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(format!("unsupported zone type: {z_type}").into());
    }

    let coords = read_coordinates(&zone, cgns_dim.phys_dim)?;
    let mut mesh = UMesh::new(coords);

    read_elements(&mut mesh, &zone)?;

    // Future implementation: read families and boundary conditions.
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mefikit::mesh::ElementLike;

    #[test]
    fn phed_stores_face_separated_node_lists() {
        let _hdf5 = crate::HDF5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        // The connectivity must end with a separator.
        assert_eq!(*conn.last().unwrap(), usize::MAX, "cell must end with separator");
    }
}
