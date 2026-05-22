use hdf5_metno::{File, Group, Dataset, types::{FixedAscii, FixedUnicode, VarLenUnicode, VarLenAscii}};
use hdf5_metno::types::TypeDescriptor;
use std::path::Path;
use ndarray::{Array1, Array2, arr1, s, array};
use mefikit::mesh::{ElementType, ElementLike, UMesh, UMeshView, Dimension};
use std::io::{self, Write};
use std::collections::BTreeMap;

pub enum CgnsHdfDtype {
    I4, // -> i32
    I8, // -> i64
    R4, // -> f32
    R8, // -> f64
    X4, // -> [f32 ; 2]or Complex<f32>
    X8, // -> [f64 ; 2] or Complex<f64>
    C1, // -> String::from_utf8
    MT, // -> No
}

// Return true if element describes a face-list section (NGON_n)
// inline to speed up the check since we'll be doing it for every element
#[inline]
fn is_ngon(cgns_code: i32) -> bool {
    cgns_code == 22
}

// Return true if element describes a cell-face section (NFACE_n)
#[inline]
fn is_nfaces(cgns_code: i32) -> bool {
    cgns_code == 23
}



fn read_string_data(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
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

fn cgns_label(group: &Group) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr(" label").or_else(|_| group.attr("label"))?;
    let label: String = attr
        .as_reader()
        .read_scalar::<FixedAscii<64>>()?
        .to_string();
    Ok(label.trim().trim_matches('\0').to_string())
}

fn find_first_child_with_label(
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

fn children_with_label(
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


fn cgns_code_to_element_type(code: i32) -> Option<ElementType> {
    match code {
        2  => Some(ElementType::VERTEX),
        3  => Some(ElementType::SEG2),
        5  => Some(ElementType::TRI3),
        7  => Some(ElementType::QUAD4),
        10 => Some(ElementType::TET4),
        17 => Some(ElementType::HEX8),
        22 => Some(ElementType::PGON),
        23 => Some(ElementType::PHED),
        other => {
            eprintln!("warning: unsupported CGNS element type {other}, section skipped");
            None
        }
    }
}

fn element_type_to_cgns(et: ElementType) -> i32 {
    match et {
        ElementType::VERTEX => 2,
        ElementType::SEG2   => 3,
        ElementType::TRI3   => 5,
        ElementType::PGON   => 22,
        ElementType::QUAD4  => 7,
        ElementType::TET4   => 10,
        ElementType::HEX8   => 17,
        ElementType::PHED   => 23,
        other => panic!("unsupported ElementType {other:?}"),
    }
}

fn nodes_per_cgns_code(code: i32) -> Option<usize> {
    match code {
        2  => Some(1),
        3  => Some(2),
        5  => Some(3),
        7  => Some(4),
        10 => Some(4),
        17 => Some(8),
        _  => None, // poly types have variable stride — handled separately
    }
}

pub fn read(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    
    Ok(())
}