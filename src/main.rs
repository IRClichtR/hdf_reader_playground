use hdf5_metno::{File, Group, Dataset, types::{FixedAscii, FixedUnicode, VarLenUnicode, VarLenAscii}};
use hdf5_metno::types::TypeDescriptor;
use std::path::Path;
use ndarray::{Array1, Array2, arr1, s, array};
use mefikit::mesh::{ElementType, ElementLike, UMesh, UMeshView, Dimension};
use std::io::{self, Write};
use std::collections::BTreeMap;

use crate::describe_dataset::describe_dataset;

mod cgns;
mod describe_dataset;

fn to_element_type(el: u8) -> ElementType {
    match el {
        1 => ElementType::VERTEX,
        3 => ElementType::SEG2,
        5 => ElementType::TRI3,
        7 => ElementType::PGON,
        9 => ElementType::QUAD4,
        10 => ElementType::TET4,
        12 => ElementType::HEX8,
        42 => ElementType::PHED,
        other => panic!("Unsupported vtk type: {other}")
    }
}

pub fn read_type_attr(group: &hdf5_metno::Group) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr("Type")?;
    let dtype = attr.dtype()?;
    let desc = dtype.to_descriptor()?;
    dbg!(&desc);  // let's see what we get
    match desc {
        TypeDescriptor::VarLenUnicode  => {
            let s: VarLenUnicode = attr.read_scalar()?;
            Ok(s.to_string())
        },
        TypeDescriptor::VarLenAscii => {
            let s: VarLenAscii = attr.read_scalar()?;
            Ok(s.to_string())
        },
        TypeDescriptor::FixedAscii(_) => {
            let s: FixedAscii<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        TypeDescriptor::FixedUnicode(_) => {
            let s: FixedUnicode<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        other => Err(format!("Unexpected string type: {other:?}").into()),
    }
}

pub fn read_type_attr2(group: &hdf5_metno::Group, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr(name)?;
    let dtype = attr.dtype()?;
    let desc = dtype.to_descriptor()?;
    dbg!(&desc);  // let's see what we get
    match desc {
        TypeDescriptor::VarLenUnicode  => {
            let s: VarLenUnicode = attr.read_scalar()?;
            Ok(s.to_string())
        },
        TypeDescriptor::VarLenAscii => {
            let s: VarLenAscii = attr.read_scalar()?;
            Ok(s.to_string())
        },
        TypeDescriptor::FixedAscii(_) => {
            let s: FixedAscii<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        TypeDescriptor::FixedUnicode(_) => {
            let s: FixedUnicode<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        other => Err(format!("Unexpected string type: {other:?}").into()),
    }
}

fn handle_unstructured(block: &hdf5_metno::Group) -> Result<UMesh, Box<dyn std::error::Error>> {
    // read data from file
    let points: Array2<f64> = block.dataset("Points")?.read()?;
    let offsets: Array1<usize> = block.dataset("Offsets")?.read()?;
    let conn: Array1<i64> = block.dataset("Connectivity")?.read()?;
    let types: Array1<usize> = block.dataset("Types")?.read()?;
    println!("points: {points:?}"); 
    println!("offsets: {offsets:?}"); 
    println!("connectivity: {conn:?}"); 
    println!("types: {types:?}"); 

    // transform data into mesh
    let mut mesh = UMesh::new(points.into());
    for i in 0..types.len() {
        let start = offsets[i];
        let end = offsets[i + 1];
        let el_type = to_element_type(types[i] as u8);
        let cell_conn: Vec<usize> = conn
            .slice(s![start..end])
            .iter()
            .map(|&x| x as usize)
            .collect();
        mesh.add_element(el_type, &cell_conn, None, None);
    }
    println!("---MESH RESULT---");
    println!("{mesh:?}");
    
    Ok(mesh)
}

fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let vtk = file.group("VTKHDF").map_err(|_| "Not a VTKHDF file")?;
    let type_attr = read_type_attr(&vtk);
    println!("type attr = {type_attr:?}");

    match read_type_attr(&vtk)?.as_str() {
        "UnstructuredGrid" => return handle_unstructured(&vtk), 
        "PartitionedDataSetCollection" | "MultiBlockDataSet" => {
            for name in vtk.member_names()? {
                let block = vtk.group(name.as_str())?;
                dbg!(&block);
                let Ok(_) = block.attr("Type") else { continue };
                match read_type_attr(&block)?.as_str() {
                    "UnstructuredGrid" => return handle_unstructured(&block),
                    _ => continue,
                }
            }
        },
        _ => {},
    }
    Err(format!("No VTKHDF group found in {}", path.display()).into())
}

fn into_vtk_u8(el: ElementType) -> u8 {
    match el {
        ElementType::VERTEX => 1,
        ElementType::SEG2   => 3,
        ElementType::TRI3   => 5,
        ElementType::PGON   => 7,
        ElementType::QUAD4  => 9,
        ElementType::TET4   => 10,
        ElementType::HEX8   => 12,
        ElementType::PHED   => 42,
        other => panic!("Unsupported ElementType {other:?}")
    }
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    // create file
    let file = File::create(path)?;
    // create VTKHDF group
    let vtk = file.create_group("VTKHDF")?;

    // add type UnstructuredGrid attr
    vtk.new_attr::<FixedAscii<16>>()
            .shape(())
            .create("Type")?
            .write_scalar(&FixedAscii::<16>::from_ascii("UnstructuredGrid").unwrap())?;
    
    // add version
    vtk.new_attr::<i64>()
        .shape([2])
        .create("Version")?
        .write(&arr1(&[2i64, 0]))?;
    
    // collect from mesh view
    let coords: Array2<f64> = mesh.coords().to_owned();

    // destructure elements of UMesh
    let mut types: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    let mut connectivity: Vec<usize> = Vec::new();

    for el in mesh.elements() {
        let conn = el.connectivity();
        types.push(into_vtk_u8(el.element_type()));
        connectivity.extend_from_slice(conn);
        offsets.push(connectivity.len());
    }
    
    // write datasets
        // coords  
    vtk.new_dataset::<f64>() 
        .shape(coords.shape())
        .create("Points")?
        .write(&coords)?;
        // types
    vtk.new_dataset::<u8>()
        .shape([types.len()])
        .create("Types")?
        .write(&Array1::from(types))?;
        // offsets
    vtk.new_dataset::<usize>() 
        .shape([offsets.len()])
        .create("Offsets")?
        .write(&Array1::from(offsets))?;
        // connectivity
    vtk.new_dataset::<usize>() 
        .shape([connectivity.len()])
        .create("Connectivity")?
        .write(&Array1::from(connectivity))?;
    
    Ok(())
}

// CGNS ===========================================================////
// 
// 
// ── primitives ────────────────────────────────────────────────────────────────


// CGNS defines two "polyhedral" element types with variable connectivity 
// length: NGON_n and NFACE_n, where n is the number of nodes per face. They 
// are encoded with cgns_code 22 and 23, respectively, and their connectivity 
// is stored as a length-prefixed list of node indices: [n_nodes, v0, v1, ..., 
// vn, n_nodes, v0, ...]. We need to handle these separately from the regular 
// fixed-stride elements.

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

// ── element type mapping ──────────────────────────────────────────────────────

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

// ── sub-readers ───────────────────────────────────────────────────────────────

fn read_coordinates(
    zone: &Group,
    phys_dim: usize,
) -> Result<ndarray::ArcArray2<f64>, Box<dyn std::error::Error>> {
    let gc = find_first_child_with_label(zone, "GridCoordinates_t")?;
    let names = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    let columns: Vec<Vec<f64>> = (0..phys_dim)
        .map(|i| {
            // CoordinateX/Y/Z are groups, data lives in their " data" dataset
            let coord_group = gc.group(names[i])
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
            let ds = coord_group.dataset(" data")
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

            // handle both R4 and R8 — check "type" attribute
            let type_attr = coord_group
                .attr("type")
                .and_then(|a| {
                    use hdf5_metno::types::FixedAscii;
                    a.as_reader().read_scalar::<FixedAscii<8>>()
                })
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "R8".to_string());

            let values: Vec<f64> = if type_attr.trim_matches('\0').starts_with("R4") {
                ds.as_reader()
                    .read_1d::<f32>()?
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            } else {
                ds.as_reader()
                    .read_1d::<f64>()?
                    .to_vec()
            };

            Ok(values)
        })
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;

    let n = columns[0].len();
    let mut coords = ndarray::Array2::<f64>::zeros((n, phys_dim));
    for (col_idx, col_data) in columns.iter().enumerate() {
        coords.column_mut(col_idx)
            .iter_mut()
            .zip(col_data)
            .for_each(|(dst, &src)| *dst = src);
    }
    let z_col = coords.column(2);
    println!("--------------------DEBUG-------------------");
    println!("Z min: {}", z_col.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("Z max: {}", z_col.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    
    Ok(coords.into_shared())
}

// first pass: collect BC ranges/pointlists → map global_cgns_idx → family_id
fn collect_bc_families(
    zone: &Group,
) -> Result<std::collections::HashMap<i64, usize>, Box<dyn std::error::Error>> {
    let mut map = std::collections::HashMap::new();

    let Ok(zonebc) = find_first_child_with_label(zone, "ZoneBC_t") else {
        return Ok(map);
    };

    for (family_id, bc) in children_with_label(&zonebc, "BC_t")?
        .into_iter()
        .enumerate()
        .map(|(i, bc)| (i + 1, bc))
    {
        let face_ids: Vec<i64> = if let Ok(pl) = bc.group("PointList") {
            pl.dataset(" data")?
                .as_reader()
                .read_dyn::<i64>()?
                .into_raw_vec_and_offset().0
        } else if let Ok(pr) = bc.group("PointRange") {
            let flat = pr.dataset(" data")?
                .as_reader()
                .read_dyn::<i64>()?
                .into_raw_vec_and_offset().0;
            (flat[0]..=flat[1]).collect()
        } else {
            continue;
        };

        for idx in face_ids {
            map.insert(idx, family_id);
        }
    }

    Ok(map)
}

fn read_elements(
    zone: &Group,
    mesh: &mut UMesh,
    bc_families: &std::collections::HashMap<i64, usize>,
) -> Result<(), Box<dyn std::error::Error>> {

    // Collect all element sections "Elements_t" and sort by starting global 
    // index (from ElementRange) to ensure we assign family tags correctly
    let mut sections = children_with_label(zone, "Elements_t")?;

    sections.sort_by_key(|s| {
        find_first_child_with_label(s, "IndexRange_t")
            .and_then(|r| r.dataset(" data")
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>))
            .and_then(|d| d.as_reader().read_dyn::<i64>()
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>))
            .map(|a| a.into_raw_vec_and_offset().0[0])
            .unwrap_or(i64::MAX)
    });

    // We run a global counter of CGNS element indices (1-based) as we read 
    // through the sections. 
    let mut global_idx = 1_i64; // 1-based running counter

    // Iterate through sections in global index order, read connectivity, 
    // add elements to mesh with family tags from bc_families map.
    for section in &sections {
        // read cgns_code from section's " data" dataset
        let meta: Vec<i32> = section
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i32>()?
            .into_raw_vec_and_offset().0;
        
        // meta[0] is cgns_code, 
        // meta[1] is parentFlag we don't use it
        let cgns_code = meta[0];

        // map cgns_code to ElementType, skip if unsupported
        // keep global_idx accurate by counting skipped elements based on 
        // ElementRange
        let Some(elem_type) = cgns_code_to_element_type(cgns_code) else {
            // count skipped elements to keep global_idx accurate
            let range: Vec<i64> = find_first_child_with_label(section, "IndexRange_t")?
                .dataset(" data")?
                .as_reader()
                .read_dyn::<i64>()?
                .into_raw_vec_and_offset().0;
            global_idx += range[1] - range[0] + 1;
            eprintln!("warning: skipping unsupported CGNS type {cgns_code}");
            continue;
        };

        // read connectivity as flat list of node indices (1-based)
        let conn: Vec<i64> = section
            .group("ElementConnectivity")?
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i64>()?
            .into_raw_vec_and_offset().0;

        // dispatch based on section type
        match nodes_per_cgns_code(cgns_code) {
            Some(stride) => {
                for chunk in conn.chunks(stride) {
                    let family = bc_families.get(&global_idx).copied();
                    let nodes: Vec<usize> = chunk
                        .iter()
                        .map(|&n| (n - 1) as usize)
                        .collect();
                    mesh.add_element(elem_type, &nodes, family, None);
                    global_idx += 1;
                }
            }

            // Format:  [n_entries, v0, …, v_{n-1},  n_entries, …]
            //
            // CRITICAL: the index-space of the values is TYPE-DEPENDENT:
            //
            //   NGON_n  (code 22) → values are 1-based NODE indices.
            //                       Decode: (v - 1) as usize.
            //
            //   NFACE_n (code 23) → values are SIGNED 1-based FACE indices.
            //                       The sign encodes face orientation; it must
            //                       NOT be stripped by subtraction before the
            //                       cast — that would wrap negatives to huge
            //                       usize values and trigger a false bounds
            //                       failure.  Use unsigned_abs() - 1 instead.
            None => {
                let mut i = 0;
                while i < conn.len() {
                    dbg!(i);
                    // Read len prefix for element
                    let n_entries = conn[i] as usize;
                    dbg!(n_entries);
                    i += 1;

                    // Guard against malformed files with incorrect 
                    // connectivity length
                    if i + n_entries > conn.len() {
                        return Err(format!(
                            "cgns_io: malformed poly connectivity in section(cgns_code={cgns_code}):  claimed {n_entries} entries at offset {} but buffer length is {}",
                            i, conn.len()
                        ).into());
                    }

                    let family = bc_families.get(&global_idx).copied();
                    let slice = &conn[i..i + n_entries];

                    let entries: Vec<usize> = if is_ngon(cgns_code) {
                        slice.iter()
                            .map(|&n| (n - 1) as usize)
                            .collect()
                    } else if is_nfaces(cgns_code) {
                        slice.iter()
                            .map(|&n| n.unsigned_abs() as usize - 1)
                            .collect()
                    } else {
                        eprintln!("Unrecognized poly cgns_code, treated as 1-based node indices: {cgns_code}");
                        slice.iter()
                            .map(|&n| (n - 1) as usize)
                            .collect()
                    };
                    
                    mesh.add_element(elem_type, &entries, family, None);
                    i += n_entries;
                    global_idx += 1;
                }
            }
        }
    }

    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

pub fn read_cgns(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;

    // base
    dbg!("Read base");
    let base = find_first_child_with_label(&file.as_group()?, "CGNSBase_t")?;
    let base_data: Vec<i32> = base
        .dataset(" data")?
        .as_reader()
        .read_dyn::<i32>()?
        .into_raw_vec_and_offset().0;
    let phys_dim = base_data[1] as usize;

    // zone
    dbg!("Read zone");
    let zone = find_first_child_with_label(&base, "Zone_t")?;

    // zone type check
    dbg!("zone type check");
    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(format!("unsupported zone type: {z_type}").into());
    }

    // coordinates
    dbg!("read coords");
    let coords = read_coordinates(&zone, phys_dim)?;
    let mut mesh = UMesh::new(coords);

    // collect BC family assignments before adding elements
    dbg!("collect BC families");
    let bc_families = collect_bc_families(&zone)?;

    // add elements with family tags already resolved
    dbg!("read elements with family tags");
    read_elements(&zone, &mut mesh, &bc_families)?;

    Ok(mesh)
}


// ── write primitives ──────────────────────────────────────────────────────────
use hdf5_metno::Datatype;
use hdf5_metno_sys::h5t::*;
use hdf5_metno_sys::h5a::*;
use hdf5_metno_sys::h5s::*;
use std::ffi::CString;

unsafe fn write_nullterm_str_attr(
    loc_id: hdf5_metno_sys::h5i::hid_t,
    attr_name: &str,
    value: &str,
    size: usize,  // 3 for "type", 33 for "label"/"name"
) -> Result<(), Box<dyn std::error::Error>> {
    // build the datatype
    unsafe {
        let tid = H5Tcopy(*H5T_C_S1);
        H5Tset_size(tid, size);
        H5Tset_strpad(tid, H5T_str_t::H5T_STR_NULLTERM);
        H5Tset_cset(tid, H5T_cset_t::H5T_CSET_ASCII);

        // scalar dataspace
        let sid = H5Screate(H5S_class_t::H5S_SCALAR);

        let attr_name_c = CString::new(attr_name)?;
        let aid = H5Acreate2(
            loc_id,
            attr_name_c.as_ptr(),
            tid,
            sid,
            hdf5_metno_sys::h5p::H5P_DEFAULT,
            hdf5_metno_sys::h5p::H5P_DEFAULT,
        );

        // write: pad value to `size` bytes, null-terminated
        let mut buf = vec![0u8; size];
        let bytes = value.as_bytes();
        buf[..bytes.len()].copy_from_slice(bytes);
        H5Awrite(aid, tid, buf.as_ptr() as *const std::ffi::c_void);

        H5Aclose(aid);
        H5Sclose(sid);
        H5Tclose(tid);
    }
    
    Ok(())
}

use hdf5_metno::Location;  // trait that exposes .id()

fn write_node_attrs(
    group: &Group,
    name: &str,
    label: &str,
    type_str: &str,
    flags: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let loc = group.id();  // hid_t
    unsafe {
        write_nullterm_str_attr(loc, "label", label, 33)?;
        write_nullterm_str_attr(loc, "name",  name,  33)?;
        write_nullterm_str_attr(loc, "type",  type_str, 3)?;
    }
    group.new_attr::<i32>()
        .shape([1])
        .create("flags")?
        .write(&ndarray::arr1(&[flags]))?;
    Ok(())
}

fn write_root_attrs(file: &File) -> Result<(), Box<dyn std::error::Error>> {
    let root = file.as_group()?;
    let loc = root.id();
    unsafe {
        write_nullterm_str_attr(loc, "label", "Root Node of HDF5 File", 33)?;
        write_nullterm_str_attr(loc, "name",  "HDF5 MotherNode",        33)?;
        write_nullterm_str_attr(loc, "type",  "MT",                     3)?;
    }
    Ok(())
}

fn write_c1_data(group: &Group, value: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<i8> = value.bytes().map(|b| b as i8).collect();
    group.new_dataset::<i8>()
        .shape([bytes.len()])
        .create(" data")?
        .write(&Array1::from(bytes))?;
    Ok(())
}

// ── write sub-functions ───────────────────────────────────────────────────────

fn write_version(file: &File) -> Result<(), Box<dyn std::error::Error>> {
    let node = file.create_group("CGNSLibraryVersion")?;
    write_node_attrs(&node, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", 0)?;
    node.new_dataset::<f32>()
        .shape([1])
        .create(" data")?
        .write(&arr1(&[3.4_f32]))?; // Write 3.4 CGNS version
    Ok(())
}

fn write_base(
    file: &File,
    cell_d: i32,
    phys_d: i32,
) -> Result<Group, Box<dyn std::error::Error>> {
    let base = file.create_group("Base")?;
    write_node_attrs(&base, "Base", "CGNSBase_t", "I4", 1)?;
    base.new_dataset::<i32>()
        .shape([2])
        .create(" data")?
        .write(&arr1(&[cell_d, phys_d]))?;
    Ok(base)
}

fn write_zone(
    base: &Group,
    n_vertices: usize,
    n_cells: usize,
) -> Result<Group, Box<dyn std::error::Error>> {
    let zone = base.create_group("Zone1")?;
    write_node_attrs(&zone, "Zone1", "Zone_t", "I8", 1)?;
    // shape [3, 1] — matches real CGNS files, required by vtkCGNSReader
    zone.new_dataset::<i32>()
        .shape([3, 1])
        .create(" data")?
        .write(&ndarray::arr2(&[
            [n_vertices as i32],
            [n_cells    as i32],
            [0_i32],
        ]))?;

    let zt = zone.create_group("ZoneType")?;
    write_node_attrs(&zt, "ZoneType", "ZoneType_t", "C1", 1)?;
    write_c1_data(&zt, "Unstructured")?;

    Ok(zone)
}

fn write_coords(zone: &Group, mesh: &UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    let gc = zone.create_group("GridCoordinates")?;
    write_node_attrs(&gc, "GridCoordinates", "GridCoordinates_t", "MT", 1)?;

    let coords = mesh.coords();
    let n      = coords.nrows();
    let names  = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    for col in 0..mesh.space_dimension() {
        let data: Vec<f64> = (0..n).map(|i| coords[[i, col]]).collect();
        let coord_node = gc.create_group(names[col])?;
        write_node_attrs(&coord_node, names[col], "DataArray_t", "R8", 0)?;
        coord_node.new_dataset::<f64>()
            .shape([n])
            .create(" data")?
            .write(&Array1::from(data))?;
    }
    Ok(())
}

fn write_elements(zone: &Group, mesh: &UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    
    // Keep track of global CGNS element index (1-based) as we write sections, 
    // to ensure correct family tag assignment in BCs.
    let mut range_start = 1_i32;

    // Iterate through blocks in mesh, group by element type, write one section per type
    for (elem_type, block) in mesh.blocks() {
        let cgns_code    = element_type_to_cgns(*elem_type);
        let n_elems      = block.len();
        let range_end    = range_start + n_elems as i32 - 1;

        // Name group with element type for readability
        let section_name = format!("{elem_type:?}");

        // section group
        let section = zone.create_group(&section_name)?;
        write_node_attrs(&section, &section_name, "Elements_t", "I4", 1)?;
        section.new_dataset::<i32>()
            .shape([2])
            .create(" data")?
            .write(&arr1(&[cgns_code, 0_i32]))?;

        // ElementRange — shape [2, 1] required by vtkCGNSReader
        let er = section.create_group("ElementRange")?;
        write_node_attrs(&er, "ElementRange", "IndexRange_t", "I8", 1)?;
        er.new_dataset::<i64>()
            .shape([2])
            .create(" data")?
            .write(&ndarray::arr1(&[range_start, range_end]))?;

        // ElementConnectivity
        let conn: Vec<i64> = match nodes_per_cgns_code(cgns_code) {
            Some(_) => {
                block.iter(mesh.coords())
                    .flat_map(|elem| {
                        elem.connectivity
                            .iter()
                            .map(|&n| (n + 1) as i64)
                    })
                    .collect()
            }
            None => {
                block.iter(mesh.coords())
                    .flat_map(|elem| {
                        let n = elem.connectivity.len() as i64;
                        std::iter::once(n)
                            .chain(elem.connectivity
                            .iter()
                            .map(|&v| (v + 1) as i64))
                        })
                        .collect()
            }
        };

        let conn_node = section.create_group("ElementConnectivity")?;
        write_node_attrs(&conn_node, "ElementConnectivity", "DataArray_t", "I8", 1)?;
        conn_node.new_dataset::<i64>()
            .shape([conn.len()])
            .create(" data")?
            .write(&Array1::from(conn))?;

        range_start = range_end + 1;
    }
    Ok(())
}

fn write_bcs(zone: &Group, mesh: &UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    // collect family_id → [global_cgns_idx, ...] (1-based)
    let mut families: BTreeMap<usize, Vec<i32>> = BTreeMap::new();
    let mut range_start = 1_i32;
    for (_, block) in mesh.blocks() {
        let n_elems = block.len();
        for (local_idx, elem) in block.iter(mesh.coords()).enumerate() {
            if *elem.family != 0 {
                let global_cgns_idx = range_start + local_idx as i32;
                families.entry(*elem.family).or_default().push(global_cgns_idx);
            }
        }
        range_start += n_elems as i32;
    }
    if families.is_empty() {
        return Ok(());
    }

    let zonebc = zone.create_group("ZoneBC")?;
    write_node_attrs(&zonebc, "ZoneBC", "ZoneBC_t", "MT", 1)?;

    for (family_id, face_ids) in &families {
        let bc_name = format!("Family_{family_id}");
        let bc = zonebc.create_group(&bc_name)?;
        write_node_attrs(&bc, &bc_name, "BC_t", "C1", 1)?;
        write_c1_data(&bc, "BCGeneral")?;

        // GridLocation
        let gl = bc.create_group("GridLocation")?;
        write_node_attrs(&gl, "GridLocation", "GridLocation_t", "C1", 1)?;
        write_c1_data(&gl, "FaceCenter")?;

        // PointList — shape [1, n]
        let n  = face_ids.len();
        let pl = bc.create_group("PointList")?;
        write_node_attrs(&pl, "PointList", "IndexArray_t", "I4", 1)?;
        pl.new_dataset::<i32>()
            .shape([n])
            .create(" data")?
            .write(&ndarray::Array1::from_shape_vec((n), face_ids.clone())?)?;
    }
    Ok(())
}


// ── entry point ─────────────────────────────────────────────────────────────

pub fn write_cgns(path: &Path, mesh: UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;

    // root group
    write_root_attrs(&file);
    
    // " format" — 15 bytes null-padded
    {
        let mut fmt = [0i8; 15];
        for (i, &b) in b"IEEE_LITTLE_32".iter().enumerate() { fmt[i] = b as i8; }
        file.new_dataset::<i8>()
            .shape([15])
            .create(" format")?
            .write(&Array1::from(fmt.to_vec()))?;
    }

    // " hdf5version" — 33 bytes null-padded
    {
        let mut ver = [0i8; 33];
        for (i, &b) in b"HDF5 Version 1.10.6".iter().enumerate() { ver[i] = b as i8; }
        file.new_dataset::<i8>()
            .shape([33])
            .create(" hdf5version")?
            .write(&Array1::from(ver.to_vec()))?;
    }

    // write CGNSLibraryVersion_t group
    write_version(&file)?;

    // Derive mesh dimensions
    // cell_dim and phys_dim are required by vtkCGNSReader. 
    // `cell_dim` is topological dimension of the volume elements (2 or 3)
    // `phys_dim` is the number of coordinate arrays in GridCoordinates_t 
    // (2 or 3)
    // they become 2 i32 values in the " data" dataset of the Base group
    let top_dim  = mesh.topological_dimension().unwrap_or(Dimension::D3);
    let cell_dim = u8::from(top_dim) as i32;
    let phys_dim = mesh.space_dimension() as i32;

    // write base group with cell_dim and phys_dim in its " data" dataset
    let base = write_base(&file, cell_dim, phys_dim)?;

    // Zone size array is [n_vertices, n_cells, 0] where:
    //   - n_cells = count of volume elements only (boundary patches excluded).
    //   - The trailing 0 is the "boundary vertex count" (0 = unspecified).
    let n_cells   = mesh.num_elements_of_dim(top_dim);
    let n_vertices = mesh.coords().nrows();
    let zone = write_zone(&base, n_vertices, n_cells)?;

    // Write coordinates
    write_coords(&zone, &mesh)?;

    // One section per block. The NGON/NFACE index-space split
    // is handled inside write_elements / encode_connectivity
    write_elements(&zone, &mesh)?;

    // Boundary conditions reference element indices by their 1-based CGNS
    // position within the contiguous ElementRange written in Element_t.
    // write_bcs rebuilds the same range_start arithmetic to ensure the
    // PointList values match.
    write_bcs(&zone, &mesh)?;

    Ok(())
}

pub fn write_roundtrip_test() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = read_cgns(Path::new("examples/cgns/particles_example.cgns"))?;
    write_cgns(Path::new("examples/cgns/roundtrip_particles8.cgns"), mesh.view())?;
    println!("wrote roundtrip_particles8.cgns");
    Ok(())
}

fn traverse_nodes(file: File) -> Result<(), Box<dyn std::error::Error>> {
    let mother = file.as_group()?;
    let main_group = find_first_child_with_label(&mother, "CGNSBase_t")?;
    let zone = find_first_child_with_label(&main_group, "Zone_t")?;
    let groups = zone.member_names()?;

    for group in groups {
        let Ok(child) = zone.group(&group) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        println!("=== Group: {group} with label {lbl} ===");

        match lbl.as_str() {
            "GridCoordinates_t" => {
                traverse_coordinates(&child)?;
            }
            "Elements_t" => {
                traverse_elements(&group, &child)?;
            }
            "ZoneBC_t" => {
                traverse_zonebc(&child)?;
            }
            _ => {}
        }
        
    }
    Ok(())
}

fn traverse_coordinates(coords_group: &Group) -> Result<(), Box<dyn std::error::Error>> {
    for name in coords_group.member_names()? {
        let Ok(child) = coords_group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        let typ = read_type_attr(&child).unwrap_or_else(|_| String::from("NO TYPE"));
        println!("  Coord: {name} | label: {lbl} | type: {typ}");
        // typ should be R4 or R8 — actual coordinate data lives
        // in the " data" dataset inside this node
        if let Ok(ds) = child.dataset(" data") {
            println!("    data shape: {:?}", ds.shape());
        }
    }
    Ok(())
}

fn traverse_elements(name: &str, elem_group: &Group) -> Result<(), Box<dyn std::error::Error>> {
    // The element type integer is the node's own data (type I4 you already see)
    // Children hold the actual arrays
    for child_name in elem_group.member_names()? {
        let Ok(child) = elem_group.group(&child_name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        let typ = read_type_attr(&child).unwrap_or_else(|_| String::from("NO TYPE"));
        println!("  [{name}] child: {child_name} | label: {lbl} | type: {typ}");
        if let Ok(ds) = child.dataset(" data") {
            println!("    data shape: {:?}", ds.shape());
        }
    }
    Ok(())
}

fn traverse_zonebc(zonebc: &Group) -> Result<(), Box<dyn std::error::Error>> {
    for patch_name in zonebc.member_names()? {
        let Ok(patch) = zonebc.group(&patch_name) else { continue };
        let Ok(lbl) = cgns_label(&patch) else { continue };
        println!("  BC patch: {patch_name} | label: {lbl}");
        for child_name in patch.member_names()? {
            let Ok(child) = patch.group(&child_name) else { continue };
            let Ok(lbl2) = cgns_label(&child) else { continue };
            let typ = read_type_attr(&child).unwrap_or_else(|_| String::from("NO TYPE"));
            println!("    child: {child_name} | label: {lbl2} | type: {typ}");
            if let Ok(ds) = child.dataset(" data") {
                println!("      data shape: {:?}", ds.shape());
            }
        }
    }
    Ok(())
}

// println!("Start");
// let files = vec![
//     "examples/cgns/yf17_hdf5.cgns",
//     "examples/cgns/particles_example.cgns",

fn main() {
    cgns::read(&Path::new("examples/cgns/yf17_hdf5.cgns")).unwrap();
    // write_roundtrip_test().unwrap();
}