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

use cubecl::{CubeScalar, num_traits::Zero, prelude::*};
use std::marker::PhantomData;
use thiserror::Error;

use crate::{data::DataBuffer, kernel::sigmoid};

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
    // stride: F,
  ) -> Result<PPResult<R, F, I>, Yolo26BcError> {
    let [n, c, s] = *pred.shape() else {
      return Err(Yolo26BcError::InvalidInputShape(
        "分类结果张量形状不正确，预期为 [N, num_classes, S]".to_string(),
      ));
    };

    tracing::debug!("输入形状: N={}, C={}, S={}", n, c, s);
    let cls: DataBuffer<R, F> = DataBuffer::with_shape(&[n, c - 4, s], client);
    let reg: DataBuffer<R, F> = DataBuffer::with_shape(&[n, 4, s], client);

    let count = (n * s).div_ceil(self.dim as usize);
    split::launch::<F, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      pred.into_tensor_arg(1),
      cls.into_tensor_arg(1),
      reg.into_tensor_arg(1),
    )?;

    let cls_sigmoid: DataBuffer<R, F> = cls.empty_like(client);

    let count = (n * c * s).div_ceil(self.dim as usize);
    sigmoid::launch::<F, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      cls.into_tensor_arg(1),
      cls_sigmoid.into_tensor_arg(1),
    )?;

    let score: DataBuffer<R, F> = DataBuffer::with_shape(&[n, s], client);
    let index: DataBuffer<R, I> = DataBuffer::with_shape(&[n, s], client);

    let count = (n * s).div_ceil(self.dim as usize);
    classify::launch::<F, I, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      cls_sigmoid.into_tensor_arg(1),
      score.into_tensor_arg(1),
      index.into_tensor_arg(1),
    )?;

    let bbox: DataBuffer<R, F> = DataBuffer::with_shape(&[n, 4, s], client);
    let stride = (self.width * self.height / s as u32) as f32;
    println!(
      "stride: {};; {:?}",
      stride.sqrt(),
      pred.shape().iter().collect::<Vec<_>>()
    );
    bbox::launch::<F, R>(
      client,
      CubeCount::Static(count as u32, 1, 1),
      CubeDim::new_1d(self.dim),
      reg.into_tensor_arg(1),
      bbox.into_tensor_arg(1),
      ScalarArg::new(F::new(self.width as f32)),
      ScalarArg::new(F::new(self.height as f32)),
      ScalarArg::new(F::new(stride.sqrt())),
    )?;

    Ok((score, index, bbox))
  }
}

/// 将 Yolo 检测结果中的分类指标进行处理，输出每个位置的最大分类得分和对应的类别索引
///
/// cls: 输入分类结果，形状为 [N, num_classes, S], 应该已经调用过 sigmoid 激活函数
/// score: 输出分类结果得分 [N, S]
/// index: 输出分类结果类型索引 [N, S]
#[cube(launch)]
fn classify<F: Float, I: Int>(cls: Tensor<F>, score: &mut Tensor<F>, index: &mut Tensor<I>) {
  // 输出张量总元素 = N * S
  let ns = cls.shape(0) * cls.shape(2);

  // 线程全局索引
  let idx = ABSOLUTE_POS;
  if idx < ns {
    // 获取输入维度
    let c_dim = cls.shape(1);
    let s_dim = cls.shape(2);

    // 将 idx 映射回 (n, s)
    // idx = n * S + s
    let n_idx = idx / s_dim;
    let s_idx = idx % s_dim;

    // 输入 strides (支持任意 stride 布局)
    let stride_n = cls.stride(0);
    let stride_c = cls.stride(1);
    let stride_s = cls.stride(2);

    // 计算 base offset (c=0 时的位置)
    let base = n_idx * stride_n + s_idx * stride_s;

    // 初始化: c=0 的值
    let mut best_c = 0;
    let mut best_val = cls[base];

    for c in 1..c_dim {
      let off = base + c * stride_c;
      let v = cls[off];
      if v > best_val {
        best_val = v;
        best_c = c;
      }
    }

    // 写入输出: 最大值 + 对应通道索引
    score[idx] = best_val;
    index[idx] = I::cast_from(best_c);
  }
}

/// 将 yolo 检测的结果进行拆分
/// 输入 pred 形状为 [N, 4 + num_classes, S]，顺序是 x, y, w, h, class_probs...
/// 输出 cls 形状为 [N, num_classes, S]，reg 形状为 [N, 4, S]
/// 输出 bbox 形状为 [N, 4, S]，包含边界框坐标 (x, y, w, h)
#[cube(launch)]
fn split<F: Float>(pred: Tensor<F>, cls: &mut Tensor<F>, reg: &mut Tensor<F>) {
  // 输出张量总元素 = N * S
  let ns = pred.shape(0) * pred.shape(2);
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

    // 拆分回归和分类结果
    for c in 0..4 {
      reg[base + c * reg.stride(1)] = pred[base + c * stride_c]; // 前4个通道是回归值
    }
    for c in 4..c_dim {
      cls[base + (c - 4) * cls.stride(1)] = pred[base + c * stride_c]; // 后续通道是分类概率
    }
  }
}

/// 将 Yolo 检测结果中的回归指标进行处理，输出每个位置的边界框坐标
/// reg: 输入回归结果，形状为 [N, 4, S], 包含 (cx, cy, w, h) 四个通道
/// bbox: 输出边界框坐标，形状为 [N, 4, S] 为 xmin, ymin, xmax, ymax
#[cube(launch)]
fn bbox<F: Float + CubeScalar + Zero>(
  reg: Tensor<F>,
  bbox: &mut Tensor<F>,
  image_width: F,
  image_height: F,
  stride: F,
) {
  let half_value = F::new(comptime!(0.5));
  let zero_value = F::new(comptime!(0.0));

  // 输出张量总元素 = N * S
  let ns = reg.shape(0) * reg.shape(2);
  let idx = ABSOLUTE_POS;

  if idx < ns {
    // 获取输入维度
    let s_dim = reg.shape(2);

    // 将 idx 映射回 (n, s)
    // idx = n * S + s
    let n_idx = idx / s_dim;
    let s_idx = idx % s_dim;

    // 输入 strides (支持任意 stride 布局)
    let stride_n = reg.stride(0);
    let stride_c = reg.stride(1);
    let stride_s = reg.stride(2);

    // 计算 base offset (c=0 时的位置)
    let base = n_idx * stride_n + s_idx * stride_s;

    let cx = reg[base]; // c=0
    let cy = reg[base + stride_c]; // c=1
    let cw = reg[base + stride_c * 2]; // c=2
    let ch = reg[base + stride_c * 3]; // c=3

    let www = image_width / stride;

    let w_idx = (F::cast_from(s_idx) % www).floor();
    let h_idx = (F::cast_from(s_idx) / www).floor();

    let grid_x = w_idx + half_value;
    let grid_y = h_idx + half_value;

    let xmin = (grid_x - cx) * stride;
    let ymin = (grid_y - cy) * stride;
    let xmax = (grid_x + cw) * stride;
    let ymax = (grid_y + ch) * stride;

    bbox[base] = xmin.clamp(zero_value, image_width); // xmin
    bbox[base + stride_c] = ymin.clamp(zero_value, image_height); // ymin
    bbox[base + 2 * stride_c] = xmax.clamp(zero_value, image_width); // xmax
    bbox[base + 3 * stride_c] = ymax.clamp(zero_value, image_height); // ymax
  }
}
