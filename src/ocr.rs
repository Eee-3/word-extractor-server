use std::error::Error;
use image::RgbImage;
use opencv::core::{AlgorithmHint, Mat, Rect};
use opencv::imgproc;
use opencv::imgproc::cvt_color;
use paddle_ocr_rs::ocr_lite::OcrLite;
use cv_convert::prelude::*;

pub fn do_ocr(image: &Mat, sorted_merged_boxes: Vec<Rect>, ocr: &mut OcrLite) -> opencv::Result<(), Box<dyn Error>> {
    for (i, &rec) in sorted_merged_boxes.iter().enumerate() {
        println!("{:?}", rec);
        let cropped_image = Mat::roi(image, rec)?;
        let mut rgb_img = Mat::default();
        cvt_color(&cropped_image, &mut rgb_img, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let image: RgbImage = rgb_img.try_to_cv()?;
        image.save(format!("e/crop_{}_i.jpg", i))?;
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