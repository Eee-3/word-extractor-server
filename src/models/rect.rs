use std::fmt::Display;
use serde::{Deserialize, Serialize};

#[derive(Debug,Copy, Clone,Serialize,Deserialize)]
pub struct Rect{
    x:u32,
    y:u32,
    width:u32,
    height:u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Rect {
        Rect {
            x,
            y,
            width,
            height,
        }
    }
}

impl From<opencv::core::Rect> for Rect{
    fn from(r:opencv::core::Rect) -> Self{
        Rect{
            x: r.x as u32,
            y: r.y as u32,
            width: r.width as u32,
            height: r.height as u32,
        }
    }
}
impl From<Rect> for opencv::core::Rect{
    fn from(r:Rect) -> Self{
        opencv::core::Rect{
            x: r.x as i32,
            y: r.y as i32,
            width: r.width as i32,
            height: r.height as i32,
        }
    }
}
impl Display for Rect{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rect: (x:{},y:{}) (width:{},height:{})", self.x, self.y, self.width, self.height)
    }
}
