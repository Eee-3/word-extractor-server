use actix_web::{HttpResponse, Responder};
use serde::Serialize;
use std::borrow::Cow;
#[derive(Serialize, Debug)]
pub struct BaseResp<'a, T: Serialize> {
    pub code: i32,
    pub message: Option<Cow<'a, str>>,
    pub data: Option<T>,
}
impl<'a, T: Serialize> BaseResp<'a, T> {
    pub fn success(data: T) -> HttpResponse {
        HttpResponse::Ok().json(BaseResp {
            code: 200,
            message: Some(Cow::from("success")),
            data: Some(data),
        })
    }
}
impl<'a> BaseResp<'a, ()> {
    pub fn error(code: i32, message: impl Into<Cow<'a, str>>) -> HttpResponse {
        HttpResponse::Ok().json(BaseResp::<'a, ()> {
            code,
            message: Some(message.into()),
            data: None,
        })
    }
}
