//! Structured Hexahedral Mesh Generation
//!
//! Generates regular grids of Hex8 elements for rectangular domains.
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// Node in 3D space.
#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub id: usize,
    pub x: Q16,
    pub y: Q16,
    pub z: Q16,
}

/// Hex8 element connectivity.
#[derive(Clone, Debug)]
pub struct Element {
    pub id: usize,
    /// Global node indices [8].
    pub nodes: [usize; 8],
}

/// Structured hexahedral mesh.
#[derive(Clone, Debug)]
pub struct HexMesh {
    pub nodes: Vec<Node>,
    pub elements: Vec<Element>,
    /// Grid divisions per axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Domain bounds.
    pub lx: Q16,
    pub ly: Q16,
    pub lz: Q16,
}

impl HexMesh {
    /// Generate structured mesh for rectangular domain [0,Lx]×[0,Ly]×[0,Lz].
    pub fn generate(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Self {
        let num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
        let num_elems = nx * ny * nz;

        let dx = Q16::from_f64(lx / nx as f64);
        let dy = Q16::from_f64(ly / ny as f64);
        let dz = Q16::from_f64(lz / nz as f64);

        // Generate nodes
        let mut nodes = Vec::with_capacity(num_nodes);
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let id = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    nodes.push(Node {
                        id,
                        x: Q16::from_int(i as i32) * dx,
                        y: Q16::from_int(j as i32) * dy,
                        z: Q16::from_int(k as i32) * dz,
                    });
                }
            }
        }

        // Generate elements
        let mut elements = Vec::with_capacity(num_elems);
        let mut eid = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n0 = k * (ny+1) * (nx+1) + j * (nx+1) + i;
                    let n1 = n0 + 1;
                    let n2 = n0 + (nx + 1) + 1;
                    let n3 = n0 + (nx + 1);
                    let n4 = n0 + (ny+1) * (nx+1);
                    let n5 = n4 + 1;
                    let n6 = n4 + (nx + 1) + 1;
                    let n7 = n4 + (nx + 1);

                    elements.push(Element {
                        id: eid,
                        nodes: [n0, n1, n2, n3, n4, n5, n6, n7],
                    });
                    eid += 1;
                }
            }
        }

        HexMesh {
            nodes, elements,
            nx, ny, nz,
            lx: Q16::from_f64(lx),
            ly: Q16::from_f64(ly),
            lz: Q16::from_f64(lz),
        }
    }

    /// Get physical coordinates of an element's 8 nodes.
    pub fn element_coords(&self, elem: &Element) -> [[Q16; 3]; 8] {
        let mut coords = [[Q16::ZERO; 3]; 8];
        for i in 0..8 {
            let n = &self.nodes[elem.nodes[i]];
            coords[i] = [n.x, n.y, n.z];
        }
        coords
    }

    /// Total DOFs (3 per node).
    pub fn num_dofs(&self) -> usize {
        self.nodes.len() * 3
    }

    /// DOF indices for a node.
    pub fn node_dofs(node_id: usize) -> [usize; 3] {
        [node_id * 3, node_id * 3 + 1, node_id * 3 + 2]
    }

    /// Find nodes on a face (x=value, y=value, or z=value).
    pub fn nodes_on_face(&self, axis: usize, value: Q16, tol: Q16) -> Vec<usize> {
        self.nodes.iter().filter_map(|n| {
            let coord = match axis {
                0 => n.x, 1 => n.y, 2 => n.z,
                _ => return None,
            };
            if (coord - value).abs().raw() <= tol.raw() {
                Some(n.id)
            } else {
                None
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_generation() {
        let mesh = HexMesh::generate(2, 2, 2, 1.0, 1.0, 1.0);
        assert_eq!(mesh.nodes.len(), 27); // 3×3×3
        assert_eq!(mesh.elements.len(), 8); // 2×2×2
        assert_eq!(mesh.num_dofs(), 81); // 27×3
    }

    #[test]
    fn test_node_positions() {
        let mesh = HexMesh::generate(2, 2, 2, 1.0, 1.0, 1.0);
        // Corner node (0,0,0) should be first
        assert_eq!(mesh.nodes[0].x.raw(), 0);
        assert_eq!(mesh.nodes[0].y.raw(), 0);
        // Corner (1,1,1) should be last
        let last = mesh.nodes.last().unwrap();
        assert!((last.x.to_f64() - 1.0).abs() < 0.01);
        assert!((last.y.to_f64() - 1.0).abs() < 0.01);
        assert!((last.z.to_f64() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_face_selection() {
        let mesh = HexMesh::generate(2, 2, 2, 1.0, 1.0, 1.0);
        let face = mesh.nodes_on_face(0, Q16::ZERO, Q16::from_raw(100));
        assert_eq!(face.len(), 9); // 3×3 nodes on x=0 face
    }
}
