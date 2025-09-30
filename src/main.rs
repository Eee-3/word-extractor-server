use opencv::imgproc::cvt_color;
use opencv::{
    Result,
    core::{CV_8UC3, Point, Rect, Scalar, Size, VecN, bitwise_and, in_range, no_array},
    highgui, imgcodecs,
    imgproc::{self, CHAIN_APPROX_SIMPLE, COLOR_BGR2HSV, RETR_EXTERNAL},
    prelude::*,
};
use std::fs;
use std::path::Path;
use opencv::core::AlgorithmHint;

fn main() -> Result<()> {
    // 读取图片
    let image = imgcodecs::imread("test/test.jpg", imgcodecs::IMREAD_COLOR)?;
    if image.empty() {
        println!("无法读取图片，请检查路径");
        return Ok(());
    }

    // 转换到 HSV 颜色空间
    let mut hsv_image = Mat::default();
    cvt_color(&image, &mut hsv_image, COLOR_BGR2HSV, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 设定绿色荧光笔的 HSV 范围
    let lower_green = Scalar::new(45.0, 50.0, 50.0, 0.0);
    let upper_green = Scalar::new(85.0, 255.0, 255.0, 0.0);

    // 根据范围创建掩码
    let mut mask = Mat::default();
    in_range(&hsv_image, &lower_green, &upper_green, &mut mask)?;

    // (可选) 对掩码进行形态学处理
    let kernel =
        imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(5, 5), Point::new(-1, -1))?;
    let mut mask_processed = Mat::default();
    imgproc::dilate(
        &mask,
        &mut mask_processed,
        &kernel,
        Point::new(-1, -1),
        2,
        opencv::core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    imgproc::erode(
        &mask_processed.clone(),
        &mut mask_processed,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // 将掩码应用到原图上
    let mut result = Mat::default();
    bitwise_and(&image, &image, &mut result, &mask_processed)?;

    // 查找轮廓
    let mut contours = opencv::core::Vector::<opencv::core::Vector<Point>>::new();
    imgproc::find_contours(
        &mask_processed,
        &mut contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut boxes = Vec::new();
    for contour in contours.iter() {
        if imgproc::contour_area(&contour, false)? > 1400.0 {
            let rect = imgproc::bounding_rect(&contour)?;
            boxes.push(rect);
        }
    }

    let merged_boxes = merge_boxes(boxes, 50.0, 20.0);

    let mut image_with_boxes = image.clone();
    let output_dir = "e/cropped";
    fs::create_dir_all(output_dir).expect("无法创建目录");

    for (i, rect) in merged_boxes.iter().enumerate() {
        imgproc::rectangle(
            &mut image_with_boxes,
            *rect,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            8,
            0,
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

    Ok(())
}

fn merge_boxes(mut boxes: Vec<Rect>, threshold_x: f64, threshold_y: f64) -> Vec<Rect> {
    if boxes.is_empty() {
        return vec![];
    }
    boxes.sort_by_key(|b| b.y);

    let mut merged_boxes = Vec::new();
    while !boxes.is_empty() {
        let mut base_box = boxes.remove(0);
        let mut i = 0;
        while i < boxes.len() {
            let other_box = boxes[i];

            let y_dist = (base_box.y.max(other_box.y)
                - (base_box.y + base_box.height).min(other_box.y + other_box.height))
                as f64;
            let x_dist = (base_box.x.max(other_box.x)
                - (base_box.x + base_box.width).min(other_box.x + other_box.width))
                as f64;

            if y_dist < threshold_y && x_dist < threshold_x {
                let min_x = base_box.x.min(other_box.x);
                let min_y = base_box.y.min(other_box.y);
                let max_x = (base_box.x + base_box.width).max(other_box.x + other_box.width);
                let max_y = (base_box.y + base_box.height).max(other_box.y + other_box.height);
                base_box.x = min_x;
                base_box.y = min_y;
                base_box.width = max_x - min_x;
                base_box.height = max_y - min_y;
                boxes.remove(i);
                i = 0; // Restart scan since base_box has changed
            } else {
                i += 1;
            }
        }
        merged_boxes.push(base_box);
    }
    merged_boxes
}
