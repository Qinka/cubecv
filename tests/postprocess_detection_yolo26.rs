// 该文件是 Shanan CV 项目的一部分。
// tests/postprocess_detection_yolo26.rs - YOLO26 后处理测试
//
// 本文件根据 Apache 许可证第 2.0 版（以下简称“许可证”）授权使用；
// 除非遵守该许可证条款，否则您不得使用本文件。
// 您可通过以下网址获取许可证副本：
// http://www.apache.org/licenses/LICENSE-2.0
// 除非适用法律要求或书面同意，根据本许可协议分发的软件均按“原样”提供，
// 不附带任何形式的明示或暗示的保证或条件。
// 有关许可权限与限制的具体条款，请参阅本许可协议。
//
// Copyright (C) 2026 Johann Li <me@qinka.pro>, Wareless Group

use std::vec;

use cubecl::prelude::*;
use shanan_cv::{data::DataBuffer, postprocess::detection::Yolo26SbcConfig};

const N: usize = 1;
const CLS: usize = 8;
const H: usize = 20;
const W: usize = 20;

#[cfg(feature = "cpu")]
#[test]
fn test_postprocess_detection_yolo26_cpu() {
  test_postprocess_detection_yolo26::<cubecl::cpu::CpuRuntime>();
}

#[cfg(feature = "wgpu")]
#[test]
fn test_postprocess_detection_yolo26_wgpu() {
  test_postprocess_detection_yolo26::<cubecl::wgpu::WgpuRuntime>();
}

fn test_postprocess_detection_yolo26<R: Runtime>() {
  let random_pred = {
    let mut v: Vec<f32> = (0..N * (1 + 4 + CLS) * H * W)
      .map(|_| rand::random::<f32>())
      .collect();

    // 设置 stride 为 32
    let stride = 32.0;
    for idx in 0..(N * (1 + 4 + CLS) * H * W) {
      let c = (idx / (H * W)) % (1 + 4 + CLS);
      if c == 0 {
        v[idx] = stride; // anchor stride
      }
    }
    v
  };

  let (score_cubecl, index_cubecl, bbox_cubecl) =
    run_postprocess_detection_yolo26_cubecl::<R>(random_pred.clone());

  let (score_manual, index_manual, bbox_manual) =
    run_postprocess_detection_yolo26_manual(random_pred, N, CLS, H * W, 640, 640);

  println!("score_cubecl\n {:?}", score_cubecl);
  println!("score_manual\n {:?}", score_manual);

  println!("index_cubecl\n {:?}", index_cubecl);
  println!("index_manual\n {:?}", index_manual);

  println!("bbox_cubecl\n {:?}", bbox_cubecl);
  println!("bbox_manual\n {:?}", bbox_manual);

  println!("box len: {} {}", bbox_cubecl.len(), bbox_manual.len());

  for (i, (s_cubecl, s_manual)) in score_cubecl.iter().zip(score_manual.iter()).enumerate() {
    assert!(
      (s_cubecl - s_manual).abs() < 1e-5,
      "得分张量第 {} 个元素不匹配: cubecl = {}, manual = {}",
      i,
      s_cubecl,
      s_manual
    );
  }

  for (i, (idx_cubecl, idx_manual)) in index_cubecl.iter().zip(index_manual.iter()).enumerate() {
    assert_eq!(
      idx_cubecl, idx_manual,
      "类别索引张量第 {} 个元素不匹配: cubecl = {}, manual = {}",
      i, idx_cubecl, idx_manual
    );
  }

  for (i, (b_cubecl, b_manual)) in bbox_cubecl.iter().zip(bbox_manual.iter()).enumerate() {
    assert!(
      (b_cubecl - b_manual).abs() < 1e-5,
      "边界框坐标张量第 {} 个元素不匹配({}): cubecl = {}, manual = {}",
      i,
      i % 400,
      b_cubecl,
      b_manual
    );
  }
}

fn run_postprocess_detection_yolo26_cubecl<R: Runtime>(
  preds: Vec<f32>,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
  let client = R::client(&R::Device::default());
  let yolo26 = Yolo26SbcConfig::default()
    .with_shape(640, 640)
    .with_dim(256)
    .build()
    .unwrap();

  let pred = DataBuffer::<R, f32>::from_slice(&preds, &[N, 5 + CLS, H * W], &client).unwrap();

  // let stride = 32.0; // 假设步幅为32
  let result = yolo26.execute(&client, &pred);
  match result {
    Ok((score, index, bbox)) => {
      println!("得分张量形状: {:?}", score.shape());
      println!("类别索引张量形状: {:?}", index.shape());
      println!("边界框坐标张量形状: {:?}", bbox.shape());

      let score = score.into_vec(&client).unwrap();
      let index = index.into_vec(&client).unwrap();
      let bbox = bbox.into_vec(&client).unwrap();

      (score, index, bbox)
    }
    Err(e) => {
      eprintln!("后处理失败: {}", e);
      panic!("后处理失败");
    }
  }
}

/// preds: [n, 4+c, s]
fn run_postprocess_detection_yolo26_manual(
  preds: Vec<f32>,
  n: usize,
  c: usize,
  s: usize,
  width: usize,
  height: usize,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
  assert_eq!(n, 1);

  let mut score_tensor = vec![0.0; n * s];
  let mut index_tensor = vec![0u32; n * s];
  let mut bbox_tensor = vec![0.0; n * 4 * s];

  let spatial = s;

  for idx in 0..s {
    let (score, class_id) = {
      let mut max_logit = f32::MIN;
      let mut cls_idx = 0usize;
      for c in 0..CLS as usize {
        let c = c + 5; // 跳过前4个回归输出
        let logit = preds[c * spatial + idx];
        if logit > max_logit {
          max_logit = logit;
          cls_idx = c - 5;
        }
      }
      (sigmoid(max_logit), cls_idx as u32)
    };

    let stride = preds[idx]; // anchor 对应的 stride
    let cx = preds[idx + 1 * spatial]; // cx
    let cy = preds[idx + 2 * spatial]; // cy
    let cw = preds[idx + 3 * spatial]; // cw
    let ch = preds[idx + 4 * spatial]; // ch

    let w = idx % W;
    let h = idx / W;

    let grid_x = (w as f32) + 0.5;
    let grid_y = (h as f32) + 0.5;

    let xmin = ((grid_x - cx) * stride).clamp(0.0, width as f32);
    let ymin = ((grid_y - cy) * stride).clamp(0.0, height as f32);
    let xmax = ((grid_x + cw) * stride).clamp(0.0, width as f32);
    let ymax = ((grid_y + ch) * stride).clamp(0.0, height as f32);

    print!(
      "idx: {}, score: {}, class_id: {}, stride: {}, bbox: ({}, {}, {}, {})\n",
      idx,
      score,
      class_id,
      stride,
      xmin / width as f32,
      ymin / height as f32,
      xmax / width as f32,
      ymax / height as f32
    );

    score_tensor[idx] = score;
    index_tensor[idx] = class_id;

    bbox_tensor[idx] = (xmin / width as f32).clamp(0.0, 1.0);
    bbox_tensor[idx + s] = (ymin / height as f32).clamp(0.0, 1.0);
    bbox_tensor[idx + 2 * s] = (xmax / width as f32).clamp(0.0, 1.0);
    bbox_tensor[idx + 3 * s] = (ymax / height as f32).clamp(0.0, 1.0);
  }

  (score_tensor, index_tensor, bbox_tensor)
}

pub fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}
