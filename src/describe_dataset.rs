use hdf5_metno::{File, Group};
use std::path::Path;

use crate::{read_type_attr2, find_first_child_with_label, cgns_label};

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
        let typ = read_type_attr2(&child, "type").unwrap_or_else(|_| String::from("NO TYPE"));
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
        let typ = read_type_attr2(&child, "type").unwrap_or_else(|_| String::from("NO TYPE"));
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
            let typ = read_type_attr2(&child, "type").unwrap_or_else(|_| String::from("NO TYPE"));
            println!("    child: {child_name} | label: {lbl2} | type: {typ}");
            if let Ok(ds) = child.dataset(" data") {
                println!("      data shape: {:?}", ds.shape());
            }
        }
    }
    Ok(())
}

pub fn describe_dataset(path: &Path) {
    println!("Start");
    let f = File::open(path).unwrap();
    traverse_nodes(f);
    // let files = vec![
    //     "examples/cgns/yf17_hdf5.cgns",
    //     "examples/cgns/particles_example.cgns",
    //     // "bump_hdf5.cgns" 
    // ];
    // for f in files {
    //     println!("!=== Reading from {f:?} ===!");
    //     // let path = Path::new(f);
    //     let f = File::open(path).unwrap();
    //     traverse_nodes(f).unwrap();
    // }
}