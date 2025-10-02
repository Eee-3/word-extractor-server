#![feature(slice_as_array)]

use actix_web::{web, App, HttpServer};
use actix_web::middleware::Logger;

mod image_process;
mod ocr;
mod models;
mod service;



#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .data(web::JsonConfig::default().limit(5012000000))
            .configure(service::register)
    })
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}



