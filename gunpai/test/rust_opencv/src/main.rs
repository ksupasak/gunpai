// use opencv::{
//     prelude::*,
//     imgcodecs,
//     highgui,
// };
//
// fn main() -> opencv::Result<()> {
//     // โหลดภาพจากไฟล์
//     let image = imgcodecs::imread("80507.jpg", imgcodecs::IMREAD_COLOR)?;
//
//     // แสดงภาพ
//     highgui::imshow("Rust OpenCV", &image)?;
//     highgui::wait_key(0)?;
//
//     Ok(())
// }

// use opencv::{
//     core,
//     dnn,
//     imgcodecs,
//     prelude::*,
//     types,
//     Result,
// };
//
// fn main() -> Result<()> {
//     // โหลดโมเดล ONNX
//     let net = dnn::read_net_from_onnx("yolov8n.onnx")?;
//     net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
//     net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
//
//     // โหลดภาพ
//     let image = imgcodecs::imread("80507.jpg", imgcodecs::IMREAD_COLOR)?;
//     let blob = dnn::blob_from_image(
//         &image,
//         1.0 / 255.0,
//         core::Size::new(640, 640), // YOLOv8 ใช้ input 640x640
//         core::Scalar::default(),
//         true,
//         false,
//         core::CV_32F,
//     )?;
//
//     // รัน model
//     net.set_input(&blob, "", 1.0, core::Scalar::default())?;
//     let mut output = types::VectorOfMat::new();
//     net.forward(&mut output, &net.get_unconnected_out_layers_names()?)?;
//
//     println!("Output layers: {}", output.len());
//     for (i, out_mat) in output.iter().enumerate() {
//         println!("Layer {} size = {:?}", i, out_mat.size()?);
//         // คุณสามารถ loop ด้านในเพื่อดูค่าต่าง ๆ ได้
//     }
//
//     Ok(())
// }

//
// use opencv::{
//     prelude::*,
//     videoio,
//     highgui,
//     imgcodecs,
//     Result,
// };
//
// fn main() -> Result<()> {
//     // เปิดกล้องตัวแรก (index 0)
//     let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
//     if !cam.is_opened()? {
//         panic!("ไม่สามารถเปิดกล้องได้");
//     }
//
//     // สร้าง Mat สำหรับเก็บภาพ
//     let mut frame = Mat::default();
//
//     loop {
//         // อ่านภาพจากกล้อง
//         cam.read(&mut frame)?;
//         if frame.empty() {
//             continue;
//         }
//
//         // แสดงผลบนหน้าจอ
//         highgui::imshow("Camera", &frame)?;
//
//         // ถ้ากด 's' ให้ save ภาพ
//         let key = highgui::wait_key(10)?;
//         if key == 115 {
//             // 115 คือ 's'
//             imgcodecs::imwrite("captured.jpg", &frame, &opencv::types::VectorOfint::new())?;
//             println!("📸 บันทึกภาพแล้วเป็น captured.jpg");
//         }
//
//         // ถ้ากด 'q' ให้ quit
//         if key == 113 {
//             // 113 คือ 'q'
//             break;
//         }
//     }
//
//     Ok(())
// }




use opencv::core;

fn main() {
    println!("OpenCV version: {}", core::CV_VERSION);
}