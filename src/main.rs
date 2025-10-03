#![feature(slice_as_array)]

use std::sync::Mutex;
use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use actix_web::middleware::Logger;
use paddle_ocr_rs::ocr_lite::OcrLite;

mod image_process;
mod ocr;
mod models;
mod service;

pub struct AppState {
    ocr: Mutex<OcrLite>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    let mut ocr = OcrLite::new();
    ocr.init_models(
        "./models/ch_PP-OCRv5_mobile_det.onnx",
        "./models/ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "./models/ch_PP-OCRv5_rec_mobile_infer.onnx",
        2,
    ).unwrap();
    let data=web::Data::new(AppState{
        ocr:Mutex::from(ocr),
    });
    HttpServer::new(move || {

        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .send_wildcard()
            .max_age(3600);
        App::new()
            .wrap(Logger::default())
            .wrap(cors)
            .data(web::JsonConfig::default().limit(10240000))
            .app_data(data.clone())
            .configure(service::register)
    })
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}



