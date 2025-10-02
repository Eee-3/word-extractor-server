use opencv::core::{AlgorithmHint, Mat, Point, Rect, Scalar, Size, bitwise_and, in_range};
use opencv::imgproc;
use opencv::imgproc::{CHAIN_APPROX_SIMPLE, COLOR_BGR2HSV, RETR_EXTERNAL, cvt_color};

pub fn detect_hl(image: &Mat) -> opencv::Result<Vec<crate::models::rect::Rect>> {
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
    Ok(sorted_merged_boxes
        .into_iter()
        .map(|r| crate::models::rect::Rect::from(r))
        .collect())
}

pub fn convert_bgr_to_hsv(image: &Mat) -> opencv::Result<Mat> {
    let mut hsv_image = Mat::default();
    cvt_color(
        &image,
        &mut hsv_image,
        COLOR_BGR2HSV,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(hsv_image)
}

pub fn sort_boxes(merged_boxes: Vec<Rect>) -> Vec<Rect> {
    let mut sorted_merged_boxes = merged_boxes;
    sorted_merged_boxes.sort_by(|a, b| {
        // 重叠
        const OVERLAP_THRESHOLD: f64 = 0.5;

        // 计算两个矩形在 Y 轴上的重叠度
        let y_min = a.y.max(b.y);
        let y_max = (a.y + a.height).min(b.y + b.height);
        let overlap_height = (y_max - y_min) as f64;
        let height_a = a.height as f64;
        let height_b = b.height as f64;
        let min_height = height_a.min(height_b);

        if min_height > 0.0 && overlap_height / min_height > OVERLAP_THRESHOLD {
            a.x.cmp(&b.x)
        } else {
            a.y.cmp(&b.y)
        }
    });
    sorted_merged_boxes
}

pub fn find_contours(mask_processed: &Mat) -> opencv::Result<Vec<Rect>> {
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
    Ok(boxes)
}

pub fn process_mask(mask: &Mat) -> opencv::Result<Mat> {
    // 形态学处理 by ai
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
    Ok(mask_processed)
}

pub fn detect_color(hsv_image: &mut Mat) -> opencv::Result<Mat> {
    // 绿色荧光笔  HSV
    let lower_green = Scalar::new(45.0, 50.0, 50.0, 0.0);
    let upper_green = Scalar::new(85.0, 255.0, 255.0, 0.0);

    let mut mask = Mat::default();
    in_range(hsv_image, &lower_green, &upper_green, &mut mask)?;
    Ok(mask)
}

pub fn merge_boxes(mut boxes: Vec<Rect>, threshold_x: f64, threshold_y: f64) -> Vec<Rect> {
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
                i = 0; // Restart scan as base_box has changed
            } else {
                i += 1;
            }
        }
        merged_boxes.push(base_box);
    }
    merged_boxes
}
