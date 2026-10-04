use ndarray::Array1;
use serde::Deserialize;
use lazy_static::lazy_static;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Point {
    pub id: u32,
    pub latitude: f64,
    pub longitude: f64,
    pub weight: f64,
    #[serde(skip)]
    pub vector: Array1<f64>,
}

impl Point {
    pub fn new(id: u32, latitude: f64, longitude: f64, weight: f64) -> Self {
        let lat_rad = latitude.to_radians();
        let lon_rad = longitude.to_radians();
        let x = super::R * lat_rad.cos() * lon_rad.cos();
        let y = super::R * lat_rad.cos() * lon_rad.sin();
        let z = super::R * lat_rad.sin();
        Self {
            id,
            latitude,
            longitude,
            weight,
            vector: Array1::from(vec![x, y, z]),
        }
    }

    pub fn index(&self) -> usize {
        // Korvatunturi has id 0, which would wrap around to a huge number if we subtract 1.
        // All other points have ids >= 2.
        // We'll map id 0 -> index 0, and id 2 -> index 1, etc.
        if self.id == 0 {
            0
        } else {
            (self.id - 1) as usize
        }
    }
}

lazy_static! {
    pub static ref KORVATUNTURI: Point = Point::new(0, 68.073611, 29.315278, 0.0);
}
