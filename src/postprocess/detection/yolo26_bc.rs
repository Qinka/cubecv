// 该文件是 Shanan CV 项目的一部分。
// src/postprocess/detection/yolo26_bc.rs
// - 基于 stride - box - class—probs 格式的 Yolo26 后处理代码
// - 尺寸应该是 [N, 1 + 4 + num_classes, S]，其中 1 是 anchor 的 stride，4 是边界框回归值，num_classes 是分类概率，
// - S 是空间位置数量，是多个”分辨率“ 下的 anchors 的总和
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

use cubecl::{CubeScalar, prelude::*};
use std::marker::PhantomData;
use thiserror::Error;

use crate::data::DataBuffer;

#[derive(Debug, Error)]
pub enum Yolo26BcError {
  #[error("无效的输入形状: {0}")]
  InvalidInputShape(String),
  #[error("运行时错误: {0}")]
  LaunchError(#[from] LaunchError),
}
pub struct Yolo26BcConfig {
  width: u32,
  height: u32,
  dim: u32,
}

impl Default for Yolo26BcConfig {
  fn default() -> Self {
    Self {
      width: 640,
      height: 640,
      dim: 1,
    }
  }
}

impl Yolo26BcConfig {
  pub fn with_shape(mut self, width: u32, height: u32) -> Self {
    self.width = width;
    self.height = height;
    self
  }

  pub fn build<R, F, I>(self) -> Result<Yolo26Bc<R, F, I>, Yolo26BcError>
  where
    R: Runtime,
    F: Float + CubeElement,
    I: Int + CubeElement,
  {
    Ok(Yolo26Bc {
      width: self.width,
      height: self.height,
      dim: self.dim,
      _phantom: PhantomData,
    })
  }

  pub fn with_dim(mut self, dim: u32) -> Self {
    self.dim = dim;
    self
  }
}

///
/// 针对 Yolo26 的后处理实现。
/// 这里，需要不同的下采样的结果分开处理（也就是 stride 得相同）
/// 然后输入尺寸应该是 B x (4 + num_classes) x S，其中 B 是批量大小，S 是所有下采样的空间位置数量之和。
/// 输出是三个张量：score (B x S)，index (B x S)，bbox (B x 4 x S)，分别是分类得分、类别索引和边界框坐标。
/// 这里没有剔除低分框的操作，后续可以在 CPU 上进行根据 score 提取 index 和 bbox 中的信息。
///
/// 对于不同的 stride 的输出，可以分别输出到 `Self::execute` 中进行处理。
/// 这里会自动计算 stride，是通过 S 和 输入图像尺寸计算得到的。(S / (width * height)).sqrt() 就是 anchor 的数量）
/// 然后需要注意的是，使用这个库的时候，请先执行 `cargo test` 来测试一下，目前发现在某些平台上 vulkan 计算结果会有问题，导致错误。
/// 然后现在 API 还在快速变化，所以，，一方面版本号最低为变动也可能导致 API 不兼容。
/// 另一方面，欢迎大家提 issue 或者 PR 来完善这个库，目前还在快速迭代中，很多功能都在开发中。
///
pub struct Yolo26Bc<R, F, I>
where
  R: Runtime,
  F: Float + CubeElement,
  I: Int + CubeElement,
{
  width: u32,
  height: u32,
  dim: u32,
  _phantom: PhantomData<(R, F, I)>,
}

pub type PPResult<R, F, I> = (DataBuffer<R, F>, DataBuffer<R, I>, DataBuffer<R, F>);

impl<R, F, I> Yolo26Bc<R, F, I>
where
  R: Runtime,
  F: Float + CubeElement,
  I: Int + CubeElement,
{
  /// 执行后处理操作
  /// pred:  YOLO 输出，形状为 [N, 4 + num_classes, N]
  /// 顺序为 x, y, w, h, class_probs...
  ///
  /// 返回 (score, index, bbox) 三个张量，分别是分类得分、类别索引和边界框坐标
  pub fn execute(
    &self,
    client: &ComputeClient<R>,
    pred: &DataBuffer<R, F>,
    threshold: F,
  ) -> Result<PPResult<R, F, I>, Yolo26BcError> {
    let [n, c, s] = *pred.shape() else {
      return Err(Yolo26BcError::InvalidInputShape(
        "分类结果张量形状不正确，预期为 [N, num_classes, S]".to_string(),
      ));
    };

    tracing::debug!("输入形状: N={}, C={}, S={}", n, c, s);
    let cls: DataBuffer<R, I> = DataBuffer::with_shape(&[n, s], client);
    let score: DataBuffer<R, F> = DataBuffer::with_shape(&[n, s], client);
    let bbox: DataBuffer<R, F> = DataBuffer::with_shape(&[n, 4, s], client);

    let stride = (self.width * self.height / s as u32) as f32;
    let count = (n * s).div_ceil(self.dim as usize);
    postprocess::launch::<F, I, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      pred.into_tensor_arg(1),
      cls.into_tensor_arg(1),
      score.into_tensor_arg(1),
      bbox.into_tensor_arg(1),
      ScalarArg::new(threshold),
      ScalarArg::new(F::new(self.width as f32)),
      ScalarArg::new(F::new(self.height as f32)),
      ScalarArg::new(F::new(stride.sqrt())),
    )?;

    Ok((score, cls, bbox))
  }
}

///
/// 将 YOLO 中的检测结果进行分类并根据结果和阈值提取 bbox
/// 输入 preds: 检测结果， 应当是 [N, (4 + num_class), S], 通道顺序为 x y w h class_probs...
/// 输出 cls: 类别 [N, S]
/// 输出 score: 类别对应的得分 [N, S]
/// 输出 bbox: 盒子坐标 [N, 4, S], 包含边界坐标 [xmin, ymin, xmax, ymax]
/// 输入 threshold：检测阈值; image_width x image_height: 图像阈值; stride: “anchor” 下采样
///
///
#[cube(launch)]
fn postprocess<F: Float + CubeScalar, I: Int>(
  pred: Tensor<F>,
  cls: &mut Tensor<I>,
  score: &mut Tensor<F>,
  bbox: &mut Tensor<F>,
  threshold: F,
  image_width: F,
  image_height: F,
  stride: F,
) {
  let one_value = F::new(comptime!(1.0));
  let half_value = F::new(comptime!(0.5));
  let zero_value = F::new(comptime!(0.0));

  let ns = pred.shape(0) * pred.shape(1);
  let idx = ABSOLUTE_POS;

  if idx < ns {
    // 获取输入维度
    let c_dim = pred.shape(1);
    let s_dim = pred.shape(2);

    // 将 idx 映射回 (n, s)
    // idx = n * S + s
    let n_idx = idx / s_dim;
    let s_idx = idx % s_dim;

    // 输入 strides (支持任意 stride 布局)
    let stride_n = pred.stride(0);
    let stride_c = pred.stride(1);
    let stride_s = pred.stride(2);

    // 计算 base offset (c=0 时的位置)
    let base = n_idx * stride_n + s_idx * stride_s;

    // 计算最佳分类
    let (best_c, best_val) = {
      let mut best_c = 4;
      let mut best_val = pred[base + 4 * stride_c];
      for c in 5..c_dim {
        let off = base + c * stride_c;
        let v = pred[off];
        if v > best_val {
          best_val = v;
          best_c = c;
        }
      }
      (best_c - 4, one_value / (one_value + (-best_val).exp()))
    };

    // 根据阈值填充矩阵并计算 bbox
    if best_val > threshold {
      // 分类结果与阈值
      cls[idx] = I::cast_from(best_c);
      score[idx] = best_val;
      // bbox
      let xmin = pred[base]; // c=0
      let ymin = pred[base + stride_c]; // c=1
      let xmax = pred[base + stride_c * 2]; // c=2
      let ymax = pred[base + stride_c * 3]; // c=3

      let www = image_width / stride;

      let w_idx = (F::cast_from(s_idx) % www).floor();
      let h_idx = (F::cast_from(s_idx) / www).floor();

      let grid_x = w_idx + half_value;
      let grid_y = h_idx + half_value;

      let xmin = (grid_x - xmin) * stride;
      let ymin = (grid_y - ymin) * stride;
      let xmax = (grid_x + xmax) * stride;
      let ymax = (grid_y + ymax) * stride;

      bbox[base] = xmin.clamp(zero_value, image_width); // xmin
      bbox[base + stride_c] = ymin.clamp(zero_value, image_height); // ymin
      bbox[base + 2 * stride_c] = xmax.clamp(zero_value, image_width); // xmax
      bbox[base + 3 * stride_c] = ymax.clamp(zero_value, image_height); // ymax
    }
  }
}
