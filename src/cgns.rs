// use hdf5_metno::{File, Group, Dataset, types::{FixedAscii, FixedUnicode, VarLenUnicode, VarLenAscii}};
// use hdf5_metno::types::TypeDescriptor;
// use std::path::Path;
// use ndarray::{Array1, Array2, arr1, s, array};
// use mefikit::mesh::{ElementType, ElementLike, UMesh, UMeshView, Dimension};
// use std::io::{self, Write};
// use std::collections::BTreeMap;
use crate::{children_with_label, describe_dataset, find_first_child_with_label, read_coordinates, read_string_data, read_type_attr2, traverse_zonebc};
use hdf5_metno::{File, Group};
use std::path::Path;
use mefikit::mesh::{ElementType, UMesh};
use indexmap::IndexSet;

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
            other  => Err(format!("Unsupported dimension combo {other:?}").into()),
        }
    }
}

struct CgnsElementInfo {
    element_type: ElementType,
    nodes_per_cell: Option<usize>,  // None for poly
}

fn cgns_element_info(code: i32) -> Option<CgnsElementInfo> {
    match code {
        2  => Some(CgnsElementInfo { element_type: ElementType::VERTEX, nodes_per_cell: Some(1) }),
        3  => Some(CgnsElementInfo { element_type: ElementType::SEG2,   nodes_per_cell: Some(2) }),
        5  => Some(CgnsElementInfo { element_type: ElementType::TRI3,   nodes_per_cell: Some(3) }),
        7  => Some(CgnsElementInfo { element_type: ElementType::QUAD4,  nodes_per_cell: Some(4) }),
        10 => Some(CgnsElementInfo { element_type: ElementType::TET4,   nodes_per_cell: Some(4) }),
        17 => Some(CgnsElementInfo { element_type: ElementType::HEX8,   nodes_per_cell: Some(8) }),
        22 => Some(CgnsElementInfo { element_type: ElementType::PGON,   nodes_per_cell: None    }),
        23 => Some(CgnsElementInfo { element_type: ElementType::PHED,   nodes_per_cell: None    }),
        other => {
            eprintln!("warning: unsupported CGNS element type {other}, section skipped");
            None
        }
    }
}

fn read_element_type(group: &Group) -> Result<i32, Box<dyn std::error::Error>> {
    let data = group
        .dataset(" data")?
        .as_reader().read_dyn::<i32>()?
            .into_raw_vec_and_offset()
            .0;

    let res = data[0];
    Ok(res)
}

fn read_index_array(group: &Group) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let ds = group.dataset(" data")?;
    let type_str = read_type_attr2(&group, "type")?;
    
    let values: Vec<i64> = match type_str.as_str() {
        "I4" => ds.as_reader().read_dyn::<i32>()?.iter().map(|&v| v as i64).collect(),
        "I8" => ds.as_reader().read_dyn::<i64>()?.into_raw_vec_and_offset().0,
        other => return Err(format!("Unexpected index type: {other}").into()),
    };
    Ok(values)
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
    // PHED can contain negative values so substract here is irrelevant
    read_index_array(&conn_group)
}

fn read_element_offsets(element: &Group) -> Result<Option<Vec<i64>>, Box<dyn std::error::Error>> {
    let Ok(offset_group) = element.group("ElementStartOffset") else {
        return Ok(None);
    };
    let values = read_index_array(&offset_group)?;
    Ok(Some(values))
}

// // ElementStartOffset = [0, 4, 9, 13, ...]
// //                       ↑  ↑  ↑   ↑
// //                       |  |  |   cell 3 starts at index 13
// //                       |  |  cell 2 starts at index 9
// //                       |  cell 1 starts at index 4
// //                       cell 0 starts at index 0

// // ElementConnectivity = [n0 n1 n2 n3 | n0 n1 n2 n3 n4 | n0 n1 n2 n3 | ...]
// //                        ←— cell 0 —→  ←——— cell 1 ———→  ←— cell 2 —→
fn read_elements(mesh: &mut UMesh, zone: &Group) -> Result<(), Box<dyn std::error::Error>> {
    let el_group = children_with_label(zone, "Elements_t")?;
    // println!("Elements_t zones: {}", el_group.len());

    // --- first pass: collect PGON and PHED raw data ---
    let mut pgon_offsets: Option<Vec<i64>> = None;
    let mut pgon_conn:    Option<Vec<i64>> = None;
    let mut phed_offsets: Option<Vec<i64>> = None;
    let mut phed_conn:    Option<Vec<i64>> = None;

    for element in &el_group {
        let type_info = cgns_element_info(read_element_type(element)?);
        if let Some(info) = type_info {
            match info.element_type {
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
                            .iter().map(|&v| v as usize).collect();
                        mesh.add_element(ElementType::PGON, &nodes, None, None);
                    }
                    pgon_offsets = Some(offsets);
                    pgon_conn    = Some(conn);
                }
                ElementType::PHED => {
                    phed_conn    = Some(read_phed_connectivity(element)?);
                    phed_offsets = Some(read_element_offsets(element)?
                        .ok_or("PHED missing ElementStartOffset")?);
                }
                other => {
                    let range = read_element_range(element)?;
                    let conn  = read_element_connectivity(element)?;
                    let n_cells = (range[1] - range[0] + 1) as usize;
                    let nodes_per_cell = info.nodes_per_cell.unwrap();
                    for i in 0..n_cells {
                        let start = i * nodes_per_cell;
                        let end   = start + nodes_per_cell;
                        let cell: Vec<usize> = conn[start..end]
                            .iter().map(|&v| v as usize).collect();
                        mesh.add_element(info.element_type, &cell, None, None);
                    }
                }
            }
        }
    }

    // --- second pass: resolve PHED using PGON ---
    if let (Some(p_off), Some(p_conn), Some(f_off), Some(f_conn)) =
        (phed_offsets, phed_conn, pgon_offsets, pgon_conn)
    {
        
        let n_cells = p_off.len() - 1;
        // println!("2nd pass n_cells: {n_cells}");
        for i in 0..n_cells {
            let start = p_off[i] as usize;
            let end   = p_off[i + 1] as usize;

            // println!("n_cell value = {i} | start = {start} | end = {end}");

            let mut cell_nodes: Vec<usize> = Vec::new();

            for &face_ref in &p_conn[start..end] {
                let _reversed  = face_ref < 0;
                let face_index = (face_ref.unsigned_abs() as usize) - 1;

                let node_start = f_off[face_index] as usize;
                let node_end   = f_off[face_index + 1] as usize;
                // println!("face_ref value = {face_ref} | node_start = {node_start} | node_end = {node_end}");

                for &node_id in &f_conn[node_start..node_end] {
                    let coord_index = (node_id as usize) - 1;
                    // println!("coord_index = {coord_index}");
                    cell_nodes.push(coord_index);
                }
            }

            mesh.add_element(ElementType::PHED, &cell_nodes, None, None);
        }
    }

    Ok(())
}

// Handle FamilyBC_t or FamilyName_t group or BC_t directly into Zone_t
// the difference is on the family name definition
fn family_name() -> Result<String, Box<dyn std::error::Error>> {
    todo!();
}

fn read_bcs(mesh: &mut UMesh, zone: &Group) -> Result<(), Box<dyn std::error::Error>> {
    let names = zone.member_names()?;
    println!("{names:?}");
    let bc = zone.group("ZoneBC")?;
    traverse_zonebc(&bc)?;

    // Handle family name get
     
    Ok(())
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    println!("<------> DATASET DESCRIPTION <------>");
    let base = find_first_child_with_label(&f.as_group()?, "CGNSBase_t")?;
    let cgns_dim = CgnsBaseDim::try_from(&base)?;
    
    println!("<------> BASE INFOS <------>");
    println!("Mesh Dimensions: {cgns_dim:?}");
    let zone = find_first_child_with_label(&base, "Zone_t")?;

    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(format!("unsupported zone type: {z_type}").into());
    }

   let coords = read_coordinates(&zone, cgns_dim.phys_dim)?;
   println!("<------> GRID COORDINATES <------>");
   // println!("coords: {coords}");
   let mut mesh = UMesh::new(coords);

   read_elements(&mut mesh, &zone)?;
   read_bcs(&mut mesh, &zone)?;
    Ok(mesh)
}