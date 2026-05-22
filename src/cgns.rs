// use hdf5_metno::{File, Group, Dataset, types::{FixedAscii, FixedUnicode, VarLenUnicode, VarLenAscii}};
// use hdf5_metno::types::TypeDescriptor;
// use std::path::Path;
// use ndarray::{Array1, Array2, arr1, s, array};
// use mefikit::mesh::{ElementType, ElementLike, UMesh, UMeshView, Dimension};
// use std::io::{self, Write};
// use std::collections::BTreeMap;
use crate::{describe_dataset, find_first_child_with_label, read_coordinates, read_string_data};
use hdf5_metno::{File, Group};
use std::path::Path;
use mefikit::mesh::{ElementType, UMesh};

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

// ElementStartOffset = [0, 4, 9, 13, ...]
//                       ↑  ↑  ↑   ↑
//                       |  |  |   cell 3 starts at index 13
//                       |  |  cell 2 starts at index 9
//                       |  cell 1 starts at index 4
//                       cell 0 starts at index 0

// ElementConnectivity = [n0 n1 n2 n3 | n0 n1 n2 n3 n4 | n0 n1 n2 n3 | ...]
//                        ←— cell 0 —→  ←——— cell 1 ———→  ←— cell 2 —→

fn read_elements() -> {
    
}

pub fn read(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    println!("<------> DATASET DESCRIPTION <------>");
    describe_dataset::describe_dataset(path);
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
   println!("coords: {coords}");
   // let mut mesh = UMesh::new(coords);
   
    Ok(())
}