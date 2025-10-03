use std::error::Error;
use image::{ImageBuffer, RgbImage};
use opencv::core::{AlgorithmHint, Mat};
use opencv::imgproc;
use opencv::imgproc::cvt_color;
use paddle_ocr_rs::ocr_lite::OcrLite;
use cv_convert::prelude::*;
use serde::de::Unexpected::Str;
use crate::models::rect::Rect;

pub fn do_ocr(image: &RgbImage, ocr: &mut OcrLite) -> opencv::Result<String, Box<dyn Error>> {
        let res = ocr.detect(
            &image,
            50,
            1024,
            0.5,
            0.3,
            1.6,
            false,
            false,
        )?;
    let mut result =String::new();
        res.text_blocks.iter().for_each(|item| {
            log::debug!("text: ({}) score: ({})", item.text, item.text_score);
            result+= &item.text;
        });
    Ok(result)
}
