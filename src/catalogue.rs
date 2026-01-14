//! Catalogue of standard Taguchi orthogonal arrays.
//!
//! This module provides easy access to standard arrays like L4, L8, L9, L18, etc.
//! using their common names. This is useful for users familiar with Taguchi methods
//! who want to reference arrays by their standard identifiers.

use crate::construct::{
    AddelmanKempthorne, Bose, BoseBush, Constructor, HadamardPaley, HadamardSylvester, RaoHamming,
};
use crate::error::{Error, Result};
use crate::oa::OA;

/// Retrieve an orthogonal array by its standard Taguchi name.
///
/// Supported arrays:
/// - **L4**: OA(4, 3, 2, 2)
/// - **L8**: OA(8, 7, 2, 2)
/// - **L9**: OA(9, 4, 3, 2)
/// - **L12**: OA(12, 11, 2, 2)
/// - **L16**: OA(16, 15, 2, 2)
/// - **L18**: OA(18, 7, 3, 2)
/// - **L25**: OA(25, 6, 5, 2)
/// - **L27**: OA(27, 13, 3, 2)
/// - **L32**: OA(32, 31, 2, 2)
/// - **L36**: OA(36, 11, 2, 2) - *Approximated by L32 or L36 not available? L36 is complex.*
///   Actually, L36 is OA(36, 13, 3, 2)? No, standard L36 is often OA(36, 2^11 3^12).
///   We will support the standard symmetric ones first.
/// - **L49**: OA(49, 8, 7, 2)
/// - **L50**: OA(50, 11, 5, 2)
/// - **L64**: OA(64, 63, 2, 2)
/// - **L81**: OA(81, 40, 3, 2)
///
/// # Example
///
/// ```
/// use taguchi::catalogue::get_by_name;
///
/// let oa = get_by_name("L9").unwrap();
/// assert_eq!(oa.runs(), 9);
/// assert_eq!(oa.levels(), 3);
/// ```
pub fn get_by_name(name: &str) -> Result<OA> {
    match name.to_uppercase().as_str() {
        "L4" => Bose::new(2).construct(3),
        "L8_BOSE" => BoseBush::new(2).map(|c| c.construct(5))?,
        // Use full capacity arrays for standard names
        "L8" => HadamardSylvester::new(8).and_then(|c| c.construct(7)),
        "L9" => Bose::new(3).construct(4),
        "L12" => HadamardPaley::new(11).and_then(|c| c.construct(11)),
        "L16" => HadamardSylvester::new(16).and_then(|c| c.construct(15)),
        "L18" => AddelmanKempthorne::new(3).and_then(|c| c.construct(7)),
        "L25" => Bose::new(5).construct(6),
        "L27" => RaoHamming::new(3, 3).and_then(|c| c.construct(13)), // L27 max cols is 13
        "L32" => HadamardSylvester::new(32).and_then(|c| c.construct(31)),
        "L49" => Bose::new(7).construct(8),
        "L50" => AddelmanKempthorne::new(5).and_then(|c| c.construct(11)),
        "L64" => HadamardSylvester::new(64).and_then(|c| c.construct(63)),
        "L81" => RaoHamming::new(3, 4).and_then(|c| c.construct(40)),
        "L128" => HadamardSylvester::new(128).and_then(|c| c.construct(127)),
        _ => Err(Error::invalid_params(format!(
            "Unknown standard array: {}",
            name
        ))),
    }
}

/// List all available standard arrays.
pub fn list_standard_arrays() -> Vec<&'static str> {
    vec![
        "L4", "L8", "L9", "L12", "L16", "L18", "L25", "L27", "L32", "L49", "L50", "L64", "L81",
        "L128",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_l9() {
        let oa = get_by_name("L9").unwrap();
        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);
    }

    #[test]
    fn test_get_l8() {
        let oa = get_by_name("L8").unwrap();
        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.levels(), 2);
    }

    #[test]
    fn test_get_l18() {
        let oa = get_by_name("L18").unwrap();
        assert_eq!(oa.runs(), 18);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.levels(), 3);
    }

    #[test]
    fn test_get_l27() {
        let oa = get_by_name("L27").unwrap();
        assert_eq!(oa.runs(), 27);
        assert_eq!(oa.factors(), 13);
        assert_eq!(oa.levels(), 3);
    }

    #[test]
    fn test_case_insensitive() {
        assert!(get_by_name("l9").is_ok());
        assert!(get_by_name("l18").is_ok());
    }

    #[test]
    fn test_unknown() {
        assert!(get_by_name("L999").is_err());
    }
}
