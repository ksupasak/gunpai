use anyhow::Result;
use opencv::{
    core::{Mat, Size, CV_8UC3},
    imgcodecs,
    imgproc,
    prelude::*,
};
use std::path::Path;

fn main() -> Result<()> {
    // อ่านรูปภาพ
    let img = imgcodecs::imread("test.jpg", imgcodecs::IMREAD_COLOR)?;
    
    // แปลงรูปภาพเป็นขนาดที่เหมาะสม
    let mut resized = Mat::default();
    imgproc::resize(
        &img,
        &mut resized,
        Size::new(640, 640),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // แปลง Mat เป็น Vec<u8> สำหรับส่งไปยัง YOLOv8
    let mut buffer = Vec::new();
    imgcodecs::imencode(".jpg", &resized, &mut buffer, &opencv::core::Vector::new())?;

    // ส่งรูปภาพไปยัง YOLOv8 API (ต้องมี YOLOv8 server ที่ทำงานอยู่)
    let client = reqwest::blocking::Client::new();
    let response = client
        .post("http://localhost:8000/predict")
        .body(buffer)
        .send()?;

    // แสดงผลลัพธ์
    println!("YOLOv8 Response: {:?}", response.text()?);

    // บันทึกรูปภาพที่ประมวลผลแล้ว
    imgcodecs::imwrite("output.jpg", &resized, &opencv::core::Vector::new())?;

    Ok(())
}
