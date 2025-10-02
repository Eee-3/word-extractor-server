mod image_process;

use image_process::*;
use opencv::{
    core::{bitwise_and, Point, Scalar},
    imgcodecs,
    imgproc::{self},
    prelude::*,
    Result,
};
use std::fs;
use std::path::Path;
use image::{RgbImage};
use cv_convert::prelude::*;
use opencv::core::AlgorithmHint;
use opencv::imgproc::cvt_color;
use paddle_ocr_rs::ocr_lite::OcrLite;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = imgcodecs::imread("test/test.jpg", imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        println!("无法读取图片，请检查路径");
        return Ok(());
    }

    let mut hsv_image = convert_bgr_to_hsv(&image)?;

    let mask = detect_color(&mut hsv_image)?;

    //形态学处理
    let mask_processed = process_mask(&mask)?;

    let mut result = Mat::default();
    bitwise_and(&image, &image, &mut result, &mask_processed)?;

    // 查找轮廓
    let boxes = find_contours(&mask_processed)?;

    let merged_boxes = merge_boxes(boxes, 50.0, 20.0);

    // 自上而下，从左到右”排序
    let sorted_merged_boxes = sort_boxes(merged_boxes);

    let mut image_with_boxes = image.clone();
    let output_dir = "e/cropped";
    fs::create_dir_all(output_dir).expect("无法创建目录");

    for (i, rect) in sorted_merged_boxes.iter().enumerate() {
        imgproc::rectangle(
            &mut image_with_boxes,
            *rect,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            8,
            0,
        )?;
        // 在矩形框旁边绘制编号
        let text = format!("{}", i + 1);
        let text_point = Point::new(rect.x, rect.y - 10); // 文字在上方
        imgproc::put_text(
            &mut image_with_boxes,
            &text,
            text_point,
            imgproc::FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar::new(0.0, 0.0, 255.0, 0.0), // 红色
            2,
            8,
            false,
        )?;
        let cropped_image = Mat::roi(&image, *rect)?;
        let output_path = Path::new(output_dir).join(format!("crop_{}.jpg", i));
        imgcodecs::imwrite(
            output_path.to_str().unwrap(),
            &cropped_image,
            &opencv::core::Vector::new(),
        )?;
    }

    fs::create_dir_all("e").expect("无法创建目录");
    imgcodecs::imwrite("e/o.jpg", &image_with_boxes, &opencv::core::Vector::new())?;
    imgcodecs::imwrite("e/m.jpg", &mask_processed, &opencv::core::Vector::new())?;
    imgcodecs::imwrite("e/r.jpg", &result, &opencv::core::Vector::new())?;

    println!("Initalizing OCR");
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv5_mobile_det.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
        2,
    )?;

    for (i,&rec) in sorted_merged_boxes.iter().enumerate(){
        println!("{:?}", rec);
        let cropped_image = Mat::roi(&image, rec)?;
        let mut rgb_img =Mat::default();
        cvt_color(&cropped_image, &mut rgb_img, imgproc::COLOR_BGR2RGB, 0,AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let image: RgbImage=rgb_img.try_to_cv()?;
        image.save(format!("e/crop_{}_i.jpg",i))?;
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
        res.text_blocks.iter().for_each(|item| {
            println!("text: ({}) score: ({})", item.text, item.text_score);
        });


    }

    Ok(())
}
