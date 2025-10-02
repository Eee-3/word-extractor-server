use serde::Deserialize;

#[derive(Debug,Deserialize)]
pub struct DetectReq{
    pub img:String ,
}