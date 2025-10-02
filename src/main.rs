#![feature(slice_as_array)]

mod image_process;
mod ocr;

use base64::engine::general_purpose::STANDARD;
use image_process::*;
use opencv::{
    imgcodecs,
    prelude::*,
    Result,
};
use paddle_ocr_rs::ocr_lite::OcrLite;
use crate::ocr::do_ocr;
use base64::prelude::*;
use image::EncodableLayout;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base64_img=std::fs::read_to_string("test/test.txt")?;
    let img_vec=STANDARD.decode(base64_img)?;


    let image = imgcodecs::imdecode(&img_vec.as_bytes(), imgcodecs::IMREAD_COLOR)?;
    // let image = imgcodecs::imread("test/test.jpg", imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        println!("无法读取图片，请检查路径");
        return Ok(());
    }

    let sorted_merged_boxes = detect_hl(&image)?;


    println!("Initalizing OCR");
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv5_mobile_det.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
        2,
    )?;

    do_ocr(&image, sorted_merged_boxes, &mut ocr)?;

    Ok(())
}



