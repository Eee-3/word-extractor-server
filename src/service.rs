use crate::image_process::detect_hl;
use crate::models::base_resp::BaseResp;
use crate::models::detect_req::DetectReq;
use crate::models::rect::Rect;
use crate::ocr::do_ocr;
use actix_web::web::ServiceConfig;
use actix_web::{HttpResponse, Responder, get, post, web};
use base64::engine::general_purpose::STANDARD;
use base64::prelude::*;
use image::EncodableLayout;
use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::prelude::*;
use paddle_ocr_rs::ocr_lite::OcrLite;
use crate::AppState;
// fn example() -> Result<(), Box<dyn std::error::Error>> {
//     let base64_img = std::fs::read_to_string("test/test.txt")?;
//     let img_vec = STANDARD.decode(base64_img)?;
//
//     let image = imgcodecs::imdecode(&Vector::<u8>::from_iter(img_vec), imgcodecs::IMREAD_COLOR)?;
//     // let image = imgcodecs::imread("test/test.jpg", imgcodecs::IMREAD_COLOR)?;
//     if image.empty() {
//         println!("无法读取图片，请检查路径");
//         return Ok(());
//     }
//
//     let sorted_merged_boxes = detect_hl(&image)?;
//
//     println!("Initalizing OCR");
//     let mut ocr = OcrLite::new();
//     ocr.init_models(
//         "./models/ch_PP-OCRv5_mobile_det.onnx",
//         "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
//         "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
//         2,
//     )?;
//
//     do_ocr(&image, sorted_merged_boxes, &mut ocr)?;
//
//     Ok(())
// }

#[get("/ping")]
async fn ping() -> impl Responder {
    BaseResp::error(-1, "pong")
}

#[post("/detect")]
async fn detect(body: web::Json<DetectReq>) -> HttpResponse {
    let img_vec = match STANDARD.decode(body.img.as_bytes()) {
        Ok(vec) => vec,
        Err(_) => return BaseResp::error(1, "图片解析错误"),
    };

    let image = match imgcodecs::imdecode(&img_vec.as_slice(), imgcodecs::IMREAD_COLOR) {
        Ok(mat) => mat,
        Err(_) => return BaseResp::error(1, "图片解码错误"),
    };

    if image.empty() {
        return BaseResp::error(1, "无法读取图片，请检查路径");
    }

    match detect_hl(&image) {
        Ok(boxes) => BaseResp::success(boxes),
        Err(e) => BaseResp::error(2, e.to_string()),
    }
}


#[post("/ocr")]
async fn ocr(body: web::Json<DetectReq>,data:web::Data<AppState>) -> HttpResponse {
    let img_vec = match STANDARD.decode(body.img.as_bytes()) {
        Ok(vec) => vec,
        Err(_) => return BaseResp::error(1, "图片解析错误"),
    };

    let image = match image::load_from_memory(&img_vec) {
        Ok(mat) => mat.into_rgb8(),
        Err(_) => return BaseResp::error(1, "图片解码错误"),
    };

    if image.is_empty() {
        return BaseResp::error(1, "无法读取图片，请检查路径");
    }
    let mut ocr = match data.ocr.lock(){
        Ok(ocr) => ocr,
        Err(_) => return BaseResp::error(3, "获取OCR锁错误"),
    };
    match do_ocr(&image, &mut ocr) {
        Ok(text) => BaseResp::success(text),
        Err(e) => BaseResp::error(4, e.to_string()),
    }
}

pub fn register(cfg: &mut ServiceConfig) {
    cfg.service(ping).service(detect).service(ocr);
}
